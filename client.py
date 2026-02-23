import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Feed Lab - Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #764ba2;
        margin-bottom: 1rem;
    }
    .upload-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .processing-header {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #764ba2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CONSTANTS
# ==============================
MODELS_DIR = "saved_models"

# ==============================
# HELPER FUNCTIONS
# ==============================

def load_saved_models():
    """Load list of saved models"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

def load_model(model_file):
    """Load a specific model"""
    try:
        model_path = os.path.join(MODELS_DIR, model_file)
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def parse_ias_5100(file):
    """Parse IAS 5100 CSV file - Extract Batch Number and spectrum data"""
    try:
        content = file.getvalue().decode('utf-8').splitlines()
        batch_number = None
        
        # Extract batch number
        for line in content[:20]:
            if 'Batch Number' in line:
                parts = line.split('\t') if '\t' in line else line.split(',')
                if len(parts) >= 2:
                    batch_number = parts[1].strip()
                    break
        
        if not batch_number:
            for line in content[:20]:
                matches = re.findall(r'B\d{6,8}(?:_\d{2})?', line)
                if matches:
                    batch_number = matches[0]
                    break
        
        if not batch_number:
            batch_number = file.name.replace('.csv', '')
        
        # Find spectrum data
        start_idx = None
        for i, line in enumerate(content):
            if 'Wavelength' in line or 'Absorbance' in line:
                start_idx = i + 1
                break
        
        if start_idx is None:
            for i, line in enumerate(content):
                parts = line.split(',')
                if len(parts) >= 2 and parts[0].replace('.', '').isdigit():
                    start_idx = i
                    break
        
        if start_idx is None:
            return None, batch_number
        
        # Parse spectrum
        wavelengths = []
        absorbances = []
        
        for line in content[start_idx:]:
            if not line.strip():
                continue
            parts = line.split(',') if ',' in line else line.split('\t')
            if len(parts) >= 2:
                try:
                    wl = float(parts[0].strip())
                    abs_val = float(parts[1].strip())
                    wavelengths.append(wl)
                    absorbances.append(abs_val)
                except:
                    continue
        
        if wavelengths:
            spectrum = pd.Series(absorbances, index=wavelengths)
            return spectrum, batch_number
        
        return None, batch_number
        
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None, file.name

def predict_with_model(model_data, spectrum):
    """Make prediction using loaded model"""
    try:
        model = model_data['model']
        model_wavelengths = [float(w) for w in model_data['wavelengths']]
        target_cols = model_data['target_cols']
        
        # Align spectrum with model wavelengths
        spectrum_aligned = spectrum.reindex(model_wavelengths)
        
        # Handle missing values
        if spectrum_aligned.isna().any():
            spectrum_aligned = spectrum_aligned.interpolate()
            spectrum_aligned = spectrum_aligned.bfill().ffill()
        
        # Prepare for prediction
        X = spectrum_aligned.values.reshape(1, -1)
        
        # Simple normalization for PLS models
        if 'PLSRegression' in str(type(model)):
            X = (X - X.mean()) / (X.std() + 1e-10)
        
        # Predict
        prediction = model.predict(X)
        prediction = np.maximum(prediction, 0)
        
        # Create result
        result = pd.DataFrame(prediction, columns=target_cols)
        return result
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_gauge(value, title, max_val=100):
    """Create a simple gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "#764ba2"},
            'steps': [
                {'range': [0, max_val/3], 'color': "#ffcccc"},
                {'range': [max_val/3, 2*max_val/3], 'color': "#ffffcc"},
                {'range': [2*max_val/3, max_val], 'color': "#ccffcc"}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ==============================
# SESSION STATE
# ==============================
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class='main-header'>
    <h1>🔮 AI Feed Lab - Predictor</h1>
    <p>Select a trained model and upload NIR files for instant predictions</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# ROW 1: Models + Upload side by side
# ==============================
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("### 📚 Available Models")
    
    # Get saved models
    saved_models = load_saved_models()
    
    if saved_models:
        # Model selection
        selected_model = st.selectbox(
            "Choose a model:",
            options=saved_models,
            format_func=lambda x: x.replace('.pkl', ''),
            key="model_selector"
        )
        
        # Load model button
        if st.button("📥 Load Selected Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                model_data = load_model(selected_model)
                if model_data:
                    st.session_state.model_data = model_data
                    st.session_state.loaded_model = model_data['model']
                    st.session_state.model_name = selected_model.replace('.pkl', '')
                    st.success(f"✅ Model loaded!")
                    st.rerun()
        
        # Show currently loaded model
        if st.session_state.model_data:
            st.markdown("---")
            st.markdown(f"""
            <div class='model-card'>
                <h4>✅ {st.session_state.model_name}</h4>
                <p><strong>Targets:</strong> {', '.join(st.session_state.model_data['target_cols'])}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No saved models found.")

with row1_col2:
    st.markdown("### 📤 Upload Files for Prediction")
    
    # Check if model is loaded
    if st.session_state.model_data is None:
        st.info("👆 Please load a model first")
    else:
        # File uploader
        uploaded_files = st.file_uploader(
            "Select IAS 5100 CSV files",
            type="csv",
            accept_multiple_files=True,
            key="file_uploader"
        )

# ==============================
# ROW 2: Results and Processing
# ==============================
if st.session_state.model_data and uploaded_files:
    # Processing header
    st.markdown("---")
    st.markdown(f"""
    <div class='processing-header'>
        <h4 style='color: #764ba2; margin: 0;'>🔄 Processing {len(uploaded_files)} file(s)...</h4>
    </div>
    """, unsafe_allow_html=True)
    
    all_predictions = []
    
    # Process each file
    for file_idx, file in enumerate(uploaded_files):
        with st.expander(f"📄 {file.name}", expanded=file_idx == 0):
            spectrum, batch_num = parse_ias_5100(file)
            
            if spectrum is not None:
                prediction = predict_with_model(st.session_state.model_data, spectrum)
                
                if prediction is not None:
                    st.markdown(f"**✅ Batch: {batch_num}**")
                    
                    # Show predictions in columns
                    target_cols = st.session_state.model_data['target_cols']
                    cols = st.columns(len(target_cols))
                    
                    for idx, target in enumerate(target_cols):
                        if target in prediction.columns:
                            value = prediction[target].iloc[0]
                            
                            # Set max values
                            if target == "Protein":
                                max_val = 50
                            elif target == "Fat":
                                max_val = 30
                            elif target == "Wa":
                                max_val = 1
                            else:
                                max_val = 10
                            
                            with cols[idx]:
                                # Create UNIQUE key for each gauge
                                gauge_key = f"gauge_{file_idx}_{idx}_{target}_{batch_num}"
                                fig = create_gauge(value, target, max_val)
                                st.plotly_chart(fig, use_container_width=True, key=gauge_key)
                    
                    # Add to summary
                    pred_row = prediction.copy()
                    pred_row['File'] = file.name
                    pred_row['Batch'] = batch_num
                    all_predictions.append(pred_row)
            else:
                st.error(f"❌ Could not parse {file.name}")
    
    # Summary section
    if all_predictions:
        st.markdown("---")
        st.markdown("### 📊 Summary of All Predictions")
        
        summary_df = pd.concat(all_predictions, ignore_index=True)
        display_cols = ['Batch', 'File'] + st.session_state.model_data['target_cols']
        display_cols = [col for col in display_cols if col in summary_df.columns]
        
        st.dataframe(summary_df[display_cols], use_container_width=True, hide_index=True)
        
        # Download button
        csv = summary_df[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions (CSV)",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="download_predictions"
        )

elif st.session_state.model_data and not uploaded_files:
    # Show upload prompt when model is loaded but no files
    st.markdown("---")
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px dashed #764ba2;
        margin: 1rem 0;
    '>
        <h3 style='color: #764ba2; margin-bottom: 1rem;'>📤 Ready for Upload</h3>
        <p style='color: #666; margin-bottom: 1rem;'>Upload files to predict:</p>
        <div style='
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
        '>
            {''.join([f"<span style='
                background: #764ba2;
                color: white;
                padding: 0.25rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
            '>{target}</span>" for target in st.session_state.model_data['target_cols']])}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "🔮 AI Feed Lab Prediction System • Client Prediction Interface"
    "</div>",
    unsafe_allow_html=True
)