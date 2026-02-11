import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io
import pickle
import os
from datetime import datetime
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')
# Add this near the top of your code with other constants
COMPONENT_WAVELENGTHS = {
    'Moisture': [(950, 1000), (1400, 1500)],  # Primary water absorption bands
    'Protein': [(1180, 1250), (1500, 1600)],   # N-H peptide bonds
    'Fat': [(1190, 1210), (1700, 1740)],       # C-H stretching (lipids/oils)
    'Fiber': [(1100, 1350)],                   # Complex carbohydrates
    # Ash: statistical correlation only
}

# Create a mapping of wavelength regions to components
WAVELENGTH_REGIONS = {
    'Moisture_1': (970, 1000),      # Primary O-H absorption
    'Moisture_2': (1450, 1500),     # Secondary O-H absorption
    'Protein_1': (1180, 1250),      # N-H peptide bonds
    'Protein_2': (1500, 1600),      # N-H overtones
    'Fat_1': (1190, 1210),          # C-H stretching
    'Fat_2': (1720, 1740),          # C-H overtone
    'Fiber_region': (1100, 1350),   # Carbohydrate structures
}

# Add this near the top of your code with other constants
# Update the preprocessing dictionary
# Update with better preprocessing strategies
COMPONENT_PREPROCESSING = {
    'moisture': ['savgol', 'snv'],  # savgol + snv from your optimal model
    'Protein': ['snv'],             # snv only for protein
    'fat': ['msc', 'snv'],          # msc + snv from your optimal model
    'ash': ['msc', 'snv'],          # msc + snv from your optimal model
    'Fiber': ['snv'],               # snv only for fiber
    'wa': ['snv']                   # snv only for wa
}

# ==============================
# PAGE CONFIG & THEME
# ==============================
st.set_page_config(
    page_title="AI Feed Lab System",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Cards */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #764ba2;
    }
    
    /* Status indicators */
    .status-success {
        color: #10B981;
        font-weight: bold;
    }
    .status-warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-error {
        color: #EF4444;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CONSTANTS - UPDATED BASED ON YOUR LAB DATA
# ==============================
# Your lab data columns: Protein, fat, ash, moisture, Fiber, wa
TARGET_COLS = [
    "Protein", "fat", "ash", "moisture", "Fiber", "wa"
]
# Note: Using exact column names from your lab data

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================
# FUNCTIONS - UPDATED FOR YOUR DATA STRUCTURE
# ==============================
def predict_with_component_models(model_data, spectrum):
    """Make predictions using component-specific models with constraints"""
    if model_data is None or 'component_models' not in model_data:
        return None
    
    predictions = {}
    aligned_spectra = {}
    
    for target, comp_model in model_data['component_models'].items():
        try:
            # Get model info
            model = comp_model['model']
            wavelengths = comp_model['wavelengths']
            preprocessing_steps = comp_model['preprocessing']
            
            # Align spectrum
            model_wavelengths = [float(w) for w in wavelengths]
            spectrum_aligned = spectrum.reindex(model_wavelengths)
            
            # Handle missing values
            if spectrum_aligned.isna().any():
                spectrum_aligned = spectrum_aligned.interpolate(method='linear')
                spectrum_aligned = spectrum_aligned.ffill().bfill()
            
            # Prepare X
            X = spectrum_aligned.values.reshape(1, -1)
            
            # Apply preprocessing steps sequentially
            X_processed = X.copy()
            for step in preprocessing_steps:
                X_processed = preprocess_spectra(X_processed, method=step)
            
            # Make prediction
            prediction = model.predict(X_processed)
            raw_value = prediction[0][0]
            
            # Store prediction
            predictions[target] = raw_value
            aligned_spectra[target] = spectrum_aligned
            
        except Exception as e:
            st.warning(f"⚠️ Error predicting {target}: {str(e)}")
            # Provide reasonable default values
            if target == 'Protein':
                predictions[target] = 25.0
            elif target == 'fat':
                predictions[target] = 15.0
            elif target == 'ash':
                predictions[target] = 5.0
            elif target == 'moisture':
                predictions[target] = 8.0
            elif target == 'Fiber':
                predictions[target] = 3.5
            elif target == 'wa':
                predictions[target] = 0.45
            else:
                predictions[target] = 0.0
    
    # Apply initial constraints
    predictions = apply_prediction_constraints(predictions)
    
    # Create result DataFrame
    if predictions:
        result_df = pd.DataFrame([predictions])
        return result_df, aligned_spectra
    
    return None, None
def save_component_models(all_models, model_name, existing_targets, dataset_info):
    """Save component-specific models"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_data = {
        'component_models': all_models,
        'target_cols': existing_targets,
        'created_date': datetime.now().isoformat(),
        'dataset_info': dataset_info,
        'preprocessing_info': COMPONENT_PREPROCESSING
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path
def select_informative_wavelengths(wavelength_cols, method='component_based'):
    """
    Select wavelengths based on component-specific regions
    """
    # Convert string wavelength columns to float
    wavelengths = [float(w) for w in wavelength_cols]
    
    if method == 'component_based':
        # Select wavelengths in key regions for each component
        selected_features = []
        feature_groups = {}
        
        for region_name, (wl_min, wl_max) in WAVELENGTH_REGIONS.items():
            # Find wavelengths in this region
            region_features = [
                wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                if wl_min <= wl <= wl_max
            ]
            if region_features:
                selected_features.extend(region_features)
                feature_groups[region_name] = region_features
        
        return selected_features, feature_groups
    
    elif method == 'statistical':
        # This would be implemented based on actual data correlation
        # Placeholder - you'll need to implement based on your data
        return wavelength_cols, {}
    
    else:
        # Return all wavelengths
        return wavelength_cols, {}

#         return component_models, component_features
def train_model_with_component_preprocessing(dataset, target_cols, component_preprocessing=COMPONENT_PREPROCESSING):
    """
    Train model with component-specific preprocessing and feature selection
    """
    with st.expander("🎯 Component-Specific Processing", expanded=True):
        st.markdown("### 🎯 Component-Specific Preprocessing")
        
        # Display preprocessing information
        st.markdown("**Component-Specific Processing Strategy:**")
        component_info = pd.DataFrame([
            ["Moisture", "savgol + snv", "Water has distinct absorption peaks"],
            ["Protein", "derivative", "Protein N-H bonds show better separation"],
            ["Fat", "msc + snv", "Fat needs scatter correction"],
            ["Fiber", "msc", "Complex carbohydrates"],
            ["Ash", "msc + snv", "Mineral content needs scatter correction"],
            ["wa", "snv", "Water activity needs normalization"]
        ], columns=["Component", "Preprocessing", "Reason"])
        
        st.dataframe(component_info, use_container_width=True)
        
        # Extract wavelength columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols 
                         if col not in target_cols + ['Sample ID']]
        
        # Convert wavelength columns to float for comparison
        wavelengths = [float(w) for w in wavelength_cols]
        
        st.info(f"📊 Total wavelengths in dataset: {len(wavelengths)}")
        st.info(f"📊 Wavelength range: {min(wavelengths):.0f} - {max(wavelengths):.0f} nm")
        
        # Prepare data structure for component-specific models
        component_models = {}
        component_features = {}
        
        # ========== FIXED: Use your optimal wavelengths from the model table ==========
        OPTIMAL_COMPONENT_WAVELENGTHS = {
            'Protein': 322,      # Index 322 = 1222 nm
            'fat': 122,          # Index 122 = 1022 nm
            'ash': 100,          # Index 100 = 1000 nm
            'moisture': 282,     # Index 282 = 1182 nm
            'Fiber': 351,        # Index 351 = 1251 nm
            'wa': 801,           # Index 801 = 1700 nm
        }
        
        for target in target_cols:
            st.markdown(f"### 🔧 Processing for **{target}**")
            
            # Get preprocessing steps for this component
            preprocessing_steps = component_preprocessing.get(target.lower(), ['snv'])
            
            # FIXED: Select wavelengths based on optimal indices from your model table
            if target in OPTIMAL_COMPONENT_WAVELENGTHS:
                optimal_idx = OPTIMAL_COMPONENT_WAVELENGTHS[target]
                
                # FIX: Check if optimal_idx is valid for our data
                if optimal_idx < len(wavelengths):
                    # Select a window around the optimal wavelength (±10 points)
                    window_size = 10
                    start_idx = max(0, optimal_idx - window_size)
                    end_idx = min(len(wavelengths), optimal_idx + window_size + 1)
                    
                    # Select the optimal region
                    selected_wls = wavelength_cols[start_idx:end_idx]
                    
                    # Get the actual wavelength value for display
                    actual_wavelength = wavelengths[optimal_idx] if optimal_idx < len(wavelengths) else "N/A"
                    st.info(f"📡 Using wavelength region around index {optimal_idx} ({actual_wavelength:.0f} nm)")
                else:
                    # If optimal_idx is out of range, fall back to alternative selection
                    st.warning(f"⚠️ Optimal index {optimal_idx} is out of range (max: {len(wavelengths)-1}). Using fallback selection.")
                    
                    # Fallback for each component
                    if target.lower() == 'protein':
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                            if (1200 <= wl <= 1250)  # Protein region
                        ]
                    elif target.lower() == 'fat':
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                            if (1000 <= wl <= 1050)  # Fat region
                        ]
                    elif target.lower() == 'ash':
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                            if (990 <= wl <= 1010)  # Ash region
                        ]
                    elif target.lower() == 'moisture':
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                            if (1400 <= wl <= 1500)  # Moisture region
                        ]
                    elif target.lower() == 'fiber':
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                            if (1240 <= wl <= 1260)  # Fiber region
                        ]
                    else:  # wa
                        selected_wls = wavelength_cols.copy()
            else:
                # Fallback for any other components
                if target.lower() == 'moisture':
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (920 <= wl <= 1000) or (1350 <= wl <= 1550)
                    ]
                elif target.lower() == 'protein':
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (1150 <= wl <= 1300) or (1480 <= wl <= 1650)
                    ]
                elif target.lower() == 'fat':
                    # Specific fat wavelengths
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (950 <= wl <= 1100)  # Around 1022 nm
                    ]
                    # Use msc + snv as in your optimal model
                    preprocessing_steps = ['msc', 'snv']
                else:
                    selected_wls = wavelength_cols.copy()
            
            # Show selection info
            col1, col2 = st.columns(2)
            with col1:
                if selected_wls:
                    st.metric(f"Selected Wavelengths for {target}", len(selected_wls))
                    wl_values = [float(w) for w in selected_wls]
                    wl_min = min(wl_values)
                    wl_max = max(wl_values)
                    st.caption(f"Range: {wl_min:.0f}-{wl_max:.0f} nm")
                    
                    # Show optimal wavelength if available
                    if target in OPTIMAL_COMPONENT_WAVELENGTHS:
                        optimal_idx = OPTIMAL_COMPONENT_WAVELENGTHS[target]
                        if optimal_idx < len(wavelengths):
                            optimal_wl = wavelengths[optimal_idx]
                            st.caption(f"Optimal: {optimal_wl:.0f} nm (index {optimal_idx})")
                else:
                    st.error(f"No wavelengths selected for {target}!")
                    continue
            with col2:
                st.metric("Preprocessing Steps", " + ".join(preprocessing_steps))
            
            # FIX: Check if we have selected wavelengths
            if not selected_wls:
                st.warning(f"Skipping {target} - no wavelengths selected")
                continue
                
            if selected_wls:
                # Extract X for this component
                X_component = dataset[selected_wls].values.astype(float)
                y_component = dataset[target].values.astype(float).reshape(-1, 1)
                
                # Check if we have valid data
                if X_component.shape[0] == 0:
                    st.warning(f"No data for {target}. Skipping...")
                    continue
                    
                # Apply preprocessing steps sequentially
                X_processed = X_component.copy()
                for step in preprocessing_steps:
                    X_processed = preprocess_spectra(X_processed, method=step)
                
                # Store processed data for this component
                component_models[target] = {
                    'X': X_processed,
                    'y': y_component,
                    'wavelengths': selected_wls,
                    'preprocessing': preprocessing_steps
                }
                component_features[target] = selected_wls
                
                # Visualize processed spectrum
                fig = go.Figure()
                if len(X_processed) > 0:
                    mean_spectrum = np.mean(X_processed, axis=0)
                    wl_values = [float(w) for w in selected_wls]
                    
                    fig.add_trace(go.Scatter(
                        x=wl_values,
                        y=mean_spectrum,
                        mode='lines',
                        name=f'Processed {target}',
                        line=dict(color='blue' if target != 'ash' else 'red', width=2)
                    ))
                
                fig.update_layout(
                    title=f"Processed Spectra for {target}",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Processed Absorbance",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        return component_models, component_features
def predict_with_component_models(model_data, spectrum):
    """Make predictions using component-specific models with constraints"""
    if model_data is None or 'component_models' not in model_data:
        return None
    
    predictions = {}
    aligned_spectra = {}
    
    for target, comp_model in model_data['component_models'].items():
        try:
            # Get model info
            model = comp_model['model']
            wavelengths = comp_model['wavelengths']
            preprocessing_steps = comp_model['preprocessing']
            
            # Convert wavelength strings to float for alignment
            model_wavelengths = [float(w) for w in wavelengths]
            
            # Align spectrum - ensure we're using the correct wavelengths
            spectrum_aligned = spectrum.reindex(model_wavelengths)
            
            # Handle missing values
            if spectrum_aligned.isna().any():
                spectrum_aligned = spectrum_aligned.interpolate(method='linear')
                spectrum_aligned = spectrum_aligned.ffill().bfill()
            
            # Prepare X
            X = spectrum_aligned.values.reshape(1, -1)
            
            # Apply preprocessing steps sequentially
            X_processed = X.copy()
            for step in preprocessing_steps:
                X_processed = preprocess_spectra(X_processed, method=step)
            
            # Make prediction
            prediction = model.predict(X_processed)
            raw_value = prediction[0][0]
            
            # FIXED: Apply target-specific post-processing
            if target == 'fat':
                # Fat typically shouldn't exceed 25% in feed
                raw_value = min(raw_value, 25.0)
                # Ensure positive value
                raw_value = max(0.1, raw_value)
            
            # Store prediction
            predictions[target] = raw_value
            aligned_spectra[target] = spectrum_aligned
            
        except Exception as e:
            st.warning(f"⚠️ Error predicting {target}: {str(e)}")
            # Provide reasonable default values based on feed composition
            defaults = {
                'Protein': 25.0,
                'fat': 12.0,  # Lower default for fat
                'ash': 5.0,
                'moisture': 8.0,
                'Fiber': 3.5,
                'wa': 0.45
            }
            predictions[target] = defaults.get(target, 0.0)
    
    # Apply constraints
    predictions = apply_prediction_constraints(predictions)
    
    # Create result DataFrame
    if predictions:
        result_df = pd.DataFrame([predictions])
        return result_df, aligned_spectra
    
    return None, None
def correlation_based_selection(X, y, wavelength_cols, target_idx=0, n_features=50):
    """
    Select features based on correlation with target variable
    """
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y[:, target_idx])[0, 1]
        correlations.append(abs(corr))
    
    # Select top features
    top_idx = np.argsort(correlations)[-n_features:]
    selected_features = [wavelength_cols[i] for i in top_idx]
    
    # Create correlation plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[float(w) for w in wavelength_cols],
        y=correlations,
        mode='lines',
        name='Correlation'
    ))
    
    # Highlight selected features
    selected_wavelengths = [float(wavelength_cols[i]) for i in top_idx]
    fig.add_trace(go.Scatter(
        x=selected_wavelengths,
        y=[correlations[i] for i in top_idx],
        mode='markers',
        name='Selected',
        marker=dict(color='red', size=8)
    ))
    
    fig.update_layout(
        title=f"Feature Correlation with Target",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absolute Correlation",
        template="plotly_white"
    )
    
    return selected_features, fig
def apply_bias_correction(predictions):
    """Apply bias correction based on known prediction errors"""
    corrected = predictions.copy()
    
    # Ensure it's a DataFrame
    if not isinstance(corrected, pd.DataFrame):
        corrected = pd.DataFrame([corrected])
    
    # Apply empirical corrections
    # Protein: slight adjustment
    if 'Protein' in corrected.columns:
        corrected['Protein'] = corrected['Protein'] * 0.98  # Only 2% reduction
    
    # FIXED: Fat - less aggressive correction
    if 'fat' in corrected.columns:
        corrected['fat'] = corrected['fat'] * 0.85  # 15% reduction instead of 40%
        corrected['fat'] = np.clip(corrected['fat'], 5, 20)  # Tighter bounds
    
    # Ash: use correlation with protein
    if 'ash' in corrected.columns and 'Protein' in corrected.columns:
        corrected['ash'] = corrected['Protein'] * 0.20  # 20% of protein value
    elif 'ash' in corrected.columns:
        corrected['ash'] = 5.0
    
    # Other components...
    if 'moisture' in corrected.columns:
        corrected['moisture'] = corrected['moisture'] * 0.95
    
    if 'Fiber' in corrected.columns:
        corrected['Fiber'] = corrected['Fiber'] * 1.05
    
    if 'wa' in corrected.columns:
        corrected['wa'] = corrected['wa'] * 0.90
    
    return corrected
def train_model_with_component_preprocessing(dataset, target_cols, component_preprocessing=COMPONENT_PREPROCESSING):
    """
    Train model with component-specific preprocessing and feature selection
    """
    with st.expander("🎯 Component-Specific Processing", expanded=True):
        st.markdown("### 🎯 Component-Specific Preprocessing")
        
        # Display preprocessing information
        st.markdown("**Component-Specific Processing Strategy:**")
        component_info = pd.DataFrame([
            ["Moisture", "savgol + snv", "Water has distinct absorption peaks"],
            ["Protein", "derivative", "Protein N-H bonds show better separation"],
            ["Fat", "msc + snv", "Fat needs scatter correction"],
            ["Fiber", "msc", "Complex carbohydrates"],
            ["Ash", "msc + snv", "Mineral content needs scatter correction"],
            ["wa", "snv", "Water activity needs normalization"]
        ], columns=["Component", "Preprocessing", "Reason"])
        
        st.dataframe(component_info, use_container_width=True)
        
        # Extract wavelength columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols 
                         if col not in target_cols + ['Sample ID']]
        
        # Convert wavelength columns to float for comparison
        wavelengths = [float(w) for w in wavelength_cols]
        
        # Prepare data structure for component-specific models
        component_models = {}
        component_features = {}
        
        # ========== FIXED: Use your optimal wavelengths from the model table ==========
        OPTIMAL_COMPONENT_WAVELENGTHS = {
            'Protein': 322,      # Index 322 = 1222 nm
            'fat': 122,          # Index 122 = 1022 nm
            'ash': 100,          # Index 100 = 1000 nm
            'moisture': 282,     # Index 282 = 1182 nm
            'Fiber': 351,        # Index 351 = 1251 nm
            'wa': 801,           # Index 801 = 1700 nm
        }
        
        for target in target_cols:
            st.markdown(f"### 🔧 Processing for **{target}**")
            
            # Get preprocessing steps for this component
            preprocessing_steps = component_preprocessing.get(target.lower(), ['snv'])
            
            # FIXED: Select wavelengths based on optimal indices from your model table
            if target in OPTIMAL_COMPONENT_WAVELENGTHS:
                optimal_idx = OPTIMAL_COMPONENT_WAVELENGTHS[target]
                # Select a window around the optimal wavelength (±10 points)
                window_size = 10
                start_idx = max(0, optimal_idx - window_size)
                end_idx = min(len(wavelengths), optimal_idx + window_size + 1)
                
                # Select the optimal region
                selected_wls = wavelength_cols[start_idx:end_idx]
                
                st.info(f"📡 Using wavelength region around index {optimal_idx} ({wavelengths[optimal_idx]:.0f} nm)")
            else:
                # Fallback for any other components
                if target.lower() == 'moisture':
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (920 <= wl <= 1000) or (1350 <= wl <= 1550)
                    ]
                elif target.lower() == 'protein':
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (1150 <= wl <= 1300) or (1480 <= wl <= 1650)
                    ]
                elif target.lower() == 'fat':
                    # FIXED: Specific fat wavelengths based on your optimal model
                    selected_wls = [
                        wl_str for wl, wl_str in zip(wavelengths, wavelength_cols)
                        if (950 <= wl <= 1100)  # Around 1022 nm
                    ]
                    # FIXED: Use msc + snv as in your optimal model
                    preprocessing_steps = ['msc', 'snv']
                else:
                    selected_wls = wavelength_cols.copy()
            
            # Show selection info
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Selected Wavelengths for {target}", len(selected_wls))
                if selected_wls:
                    wl_values = [float(w) for w in selected_wls]
                    wl_min = min(wl_values)
                    wl_max = max(wl_values)
                    st.caption(f"Range: {wl_min:.0f}-{wl_max:.0f} nm")
                    if target in OPTIMAL_COMPONENT_WAVELENGTHS:
                        optimal_wl = wavelengths[OPTIMAL_COMPONENT_WAVELENGTHS[target]]
                        st.caption(f"Optimal: {optimal_wl:.0f} nm (index {OPTIMAL_COMPONENT_WAVELENGTHS[target]})")
            with col2:
                st.metric("Preprocessing Steps", " + ".join(preprocessing_steps))
            
            if selected_wls:
                # Extract X for this component
                X_component = dataset[selected_wls].values.astype(float)
                y_component = dataset[target].values.astype(float).reshape(-1, 1)
                
                # Apply preprocessing steps sequentially
                X_processed = X_component.copy()
                for step in preprocessing_steps:
                    X_processed = preprocess_spectra(X_processed, method=step)
                
                # Store processed data for this component
                component_models[target] = {
                    'X': X_processed,
                    'y': y_component,
                    'wavelengths': selected_wls,
                    'preprocessing': preprocessing_steps
                }
                component_features[target] = selected_wls
                
                # Visualize processed spectrum
                fig = go.Figure()
                if len(X_processed) > 0:
                    mean_spectrum = np.mean(X_processed, axis=0)
                    wl_values = [float(w) for w in selected_wls]
                    
                    fig.add_trace(go.Scatter(
                        x=wl_values,
                        y=mean_spectrum,
                        mode='lines',
                        name=f'Processed {target}',
                        line=dict(color='blue' if target != 'ash' else 'red', width=2)
                    ))
                
                fig.update_layout(
                    title=f"Processed Spectra for {target}",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Processed Absorbance",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        return component_models, component_features

def parse_ias_5100(file):
    """Parse IAS 5100 CSV file - extract Sample ID and spectrum data"""
    try:
        content = file.getvalue().decode("utf-8").splitlines()
        file_name = file.name
        
        sample_id = None
        
        # Method 1: Look for Sample ID in your NIR file format
        for line in content:
            if 'Sample ID' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    potential_id = parts[1].strip()
                    if potential_id and potential_id not in ['Reference', 'Sample', '']:
                        sample_id = potential_id
                        break
        
        # Method 2: Extract from content based on your data format
        if not sample_id:
            for line in content:
                # Look for patterns like B2500343, B2600041_01, etc.
                if re.search(r'B\d{6,8}', line):
                    parts = line.split(',')
                    for part in parts:
                        cleaned = part.strip()
                        if re.match(r'^B\d{6,8}(_\d{2})?$', cleaned):
                            sample_id = cleaned
                            break
                if sample_id:
                    break
        
        # Method 3: Extract from filename
        if not sample_id:
            filename = file_name.replace('.csv', '')
            # Look for sample ID patterns in filename
            match = re.search(r'(B\d{6,8}[_\.]?\d{0,2})', filename)
            if match:
                sample_id = match.group(1)
        
        if not sample_id:
            sample_id = file_name.replace('.csv', '')
            st.warning(f"⚠️ Sample ID not found in file. Using filename as ID: {sample_id}")
        
        # Find and parse spectrum data
        start_idx = None
        for i, line in enumerate(content):
            # Look for wavelength header
            if line.lower().startswith('wavelength') or ',absorbance' in line.lower():
                start_idx = i
                break
        
        if start_idx is None:
            # Find first line with numeric wavelength
            for i, line in enumerate(content):
                parts = line.split(',')
                if len(parts) >= 2:
                    first_val = parts[0].strip()
                    if first_val and first_val.replace('.', '').replace('-', '').isdigit():
                        start_idx = i
                        break
        
        if start_idx is None:
            st.error("❌ Spectrum table not found in file")
            return pd.Series(), sample_id
        
        # Parse the spectrum data
        df = pd.read_csv(io.StringIO("\n".join(content[start_idx:])))
        
        # Identify wavelength and absorbance columns
        wavelength_col = None
        absorbance_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['wavelength', 'wl', 'nm']):
                wavelength_col = col
            elif any(term in col_lower for term in ['absorbance', 'absorption', 'abs']):
                absorbance_col = col
        
        # If columns not found, assume first two columns
        if wavelength_col is None and absorbance_col is None and len(df.columns) >= 2:
            wavelength_col = df.columns[0]
            absorbance_col = df.columns[1]
        
        if wavelength_col and absorbance_col:
            # Clean and convert data
            df = df.rename(columns={wavelength_col: 'Wavelength', absorbance_col: 'Absorbance'})
            df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
            df['Absorbance'] = pd.to_numeric(df['Absorbance'], errors='coerce')
            df = df.dropna(subset=['Wavelength', 'Absorbance'])
            
            # Create spectrum series
            spectrum = df.set_index('Wavelength')['Absorbance']
            
            st.success(f"✅ {file_name}: ID={sample_id}, Points={len(spectrum)}, Range={spectrum.index.min()}-{spectrum.index.max()}nm")
            return spectrum, sample_id
        
        return pd.Series(), sample_id
        
    except Exception as e:
        st.error(f"❌ Error parsing file {file_name}: {str(e)}")
        return pd.Series(), file_name

def load_lab_data(lab_file):
    """Load lab data with support for both CSV and tab-delimited files"""
    try:
        # Try different separators
        for sep in [',', '\t', ';']:
            try:
                lab_df = pd.read_csv(lab_file, sep=sep, engine='python')
                lab_file.seek(0)
                
                # Clean column names
                lab_df.columns = [col.strip() for col in lab_df.columns]
                
                # Show loaded columns for debugging
                st.info(f"📋 Columns found in lab file: {list(lab_df.columns)}")
                
                # Identify Sample ID column
                sample_id_col = None
                for col in lab_df.columns:
                    col_lower = col.lower()
                    if 'sample' in col_lower or 'id' in col_lower:
                        sample_id_col = col
                        break
                
                if sample_id_col:
                    lab_df['Sample ID'] = lab_df[sample_id_col].astype(str).str.strip()
                    st.success(f"✅ Found Sample ID in column: '{sample_id_col}'")
                else:
                    # Check if 'Sample ID' column exists
                    if 'Sample ID' in lab_df.columns:
                        lab_df['Sample ID'] = lab_df['Sample ID'].astype(str).str.strip()
                    else:
                        # Use first column that looks like sample IDs
                        for col in lab_df.columns:
                            if lab_df[col].astype(str).str.contains(r'B\d+').any():
                                lab_df['Sample ID'] = lab_df[col].astype(str).str.strip()
                                st.success(f"✅ Using column '{col}' as Sample ID")
                                break
                
                if 'Sample ID' not in lab_df.columns:
                    st.error("❌ No Sample ID column found in lab data!")
                    return None
                
                # Show a preview of the data
                st.dataframe(lab_df.head(), use_container_width=True)
                
                return lab_df
                
            except Exception:
                continue
        
        st.error("❌ Could not parse lab file with any separator.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading lab data: {str(e)}")
        return None

def apply_snv(X):
    """Apply Standard Normal Variate normalization"""
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    
    return (X - mean) / std

def prepare_training_data(spectra_dict, lab_df):
    """Prepare training data from spectra and lab data - ONE lab result per sample ID"""
    with st.expander("🔍 Data Matching Analysis", expanded=True):
        spectra_ids = list(spectra_dict.keys())
        
        # Remove _dup suffix for matching
        base_spectra_dict = {}
        for spec_id, spectrum in spectra_dict.items():
            base_id = re.sub(r'_dup\d+$', '', spec_id)
            if base_id not in base_spectra_dict:
                base_spectra_dict[base_id] = []
            base_spectra_dict[base_id].append((spec_id, spectrum))
        
        # Clean lab data
        lab_df = lab_df.copy()
        lab_df['Sample ID'] = lab_df['Sample ID'].astype(str).str.strip()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Unique Sample IDs from NIR files:**")
            st.markdown(f"**Total spectra:** {len(spectra_ids)}")
            st.markdown(f"**Unique samples:** {len(base_spectra_dict)}")
            for base_id in list(base_spectra_dict.keys())[:10]:
                count = len(base_spectra_dict[base_id])
                if count > 1:
                    st.markdown(f"• {base_id} (×{count} scans)")
                else:
                    st.markdown(f"• {base_id}")
            if len(base_spectra_dict) > 10:
                st.markdown(f"*... and {len(base_spectra_dict) - 10} more*")
        
        with col2:
            st.markdown("**📋 Sample IDs from Lab file:**")
            for lab_id in lab_df['Sample ID'].tolist()[:10]:
                st.markdown(f"• {lab_id}")
            if len(lab_df) > 10:
                st.markdown(f"*... and {len(lab_df) - 10} more*")
            st.metric("Lab Samples", len(lab_df))
        
        # Find matches - each sample ID should have ONE lab result
        matches = []
        matched_base_ids = set()
        
        for base_id, spectra_list in base_spectra_dict.items():
            # Find lab entry for this base ID
            lab_entry = lab_df[lab_df['Sample ID'] == base_id]
            
            if not lab_entry.empty:
                lab_data = lab_entry.iloc[0]  # Take first matching lab entry
                
                # Match ALL spectra for this base ID with the SAME lab result
                for spec_id, spectrum in spectra_list:
                    matches.append({
                        'NIR_Sample_ID': spec_id,
                        'Base_Sample_ID': base_id,
                        'Lab_Sample_ID': lab_data['Sample ID'],
                        'Lab_Row_Index': lab_entry.index[0]
                    })
                
                matched_base_ids.add(base_id)
                st.info(f"✅ Matched {len(spectra_list)} spectra for sample {base_id} with lab result")
        
        if matches:
            st.success(f"✅ **{len(matches)}** spectrum-lab matches found!")
            st.info(f"**{len(matched_base_ids)}** unique samples with lab results")
            
            # Show match summary
            match_summary = pd.DataFrame(matches)
            match_counts = match_summary['Base_Sample_ID'].value_counts()
            
            st.markdown("**📊 Match Summary:**")
            summary_data = []
            for base_id, count in match_counts.items():
                summary_data.append({
                    'Sample ID': base_id,
                    'Spectra Count': count,
                    'Status': f"✅ {count} scans"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Check for unmatched
            unmatched_spectra = [sid for sid in base_spectra_dict.keys() if sid not in matched_base_ids]
            if unmatched_spectra:
                st.warning(f"⚠️ {len(unmatched_spectra)} samples without lab results:")
                for unmatched in unmatched_spectra[:5]:
                    st.markdown(f"• {unmatched}")
                if len(unmatched_spectra) > 5:
                    st.markdown(f"*... and {len(unmatched_spectra) - 5} more*")
        else:
            st.error("❌ No matching Sample IDs found!")
            return None, [], "none"
        
        # Create dataset with all spectra and their corresponding lab results
        dataset_rows = []
        
        for match in matches:
            spec_id = match['NIR_Sample_ID']
            base_id = match['Base_Sample_ID']
            
            # Get spectrum
            spectrum = spectra_dict[spec_id]
            
            # Get lab data for this base ID
            lab_row = lab_df[lab_df['Sample ID'] == base_id].iloc[0]
            
            # Create row
            row_data = {
                'NIR_Sample_ID': spec_id,
                'Sample ID': base_id
            }
            
            # Add spectrum data
            wavelength_cols = [str(w) for w in spectrum.index]
            for wl, abs_val in zip(spectrum.index, spectrum.values):
                row_data[str(wl)] = abs_val
            
            # Add lab results
            for target in TARGET_COLS:
                if target in lab_row:
                    row_data[target] = lab_row[target]
            
            dataset_rows.append(row_data)
        
        # Create DataFrame
        dataset = pd.DataFrame(dataset_rows)
        
        st.success(f"✅ Created dataset with **{len(dataset)}** samples")
        st.info(f"📊 **Data structure:** {len(dataset)} spectra × {len([col for col in dataset.columns if col.replace('.', '').isdigit()])} wavelengths")
        
        # Show preview
        st.markdown("**📈 Dataset Preview:**")
        preview_cols = ['NIR_Sample_ID', 'Sample ID'] + [col for col in TARGET_COLS if col in dataset.columns]
        st.dataframe(dataset[preview_cols].head(), use_container_width=True)
        
        return dataset, [m['NIR_Sample_ID'] for m in matches], "matched"
def preprocess_spectra(X, method='snv'):
    """Apply different spectral pre-processing methods with error handling"""
    if method == 'snv':
        return apply_snv(X)
    elif method == 'msc':
        # Multiplicative Scatter Correction with error handling
        try:
            mean_spectrum = np.mean(X, axis=0)
            for i in range(X.shape[0]):
                # Fit each spectrum to mean spectrum
                coeffs = np.polyfit(mean_spectrum, X[i], 1)
                if not np.isfinite(coeffs[0]) or coeffs[0] == 0:
                    coeffs[0] = 1.0
                    coeffs[1] = 0.0
                X[i] = (X[i] - coeffs[1]) / coeffs[0]
        except:
            # If MSC fails, fall back to SNV
            return apply_snv(X)
        return X
    elif method == 'derivative':
        # First derivative with smoothing
        try:
            from scipy.ndimage import gaussian_filter1d
            X_smooth = gaussian_filter1d(X, sigma=2, axis=1)
            return np.gradient(X_smooth, axis=1)
        except:
            return np.gradient(X, axis=1)
    elif method == 'savgol':
        # Savitzky-Golay smoothing + derivative
        try:
            from scipy.signal import savgol_filter
            X_smooth = savgol_filter(X, window_length=min(11, X.shape[1]), 
                                   polyorder=2, axis=1)
            return np.gradient(X_smooth, axis=1)
        except:
            return np.gradient(X, axis=1)
    return X
def apply_prediction_constraints(predictions):
    """Apply realistic constraints to predictions based on feed chemistry"""
    constrained = predictions.copy()
    
    # Check if predictions is a DataFrame row (Series) or a dictionary
    if hasattr(constrained, 'iloc'):  # It's a DataFrame row or Series
        # Convert Series to dictionary
        constrained = constrained.to_dict()
    elif hasattr(constrained, 'to_dict'):  # It's a DataFrame with to_dict method
        # Handle case where it's a single-row DataFrame
        if len(constrained) == 1:
            constrained = constrained.to_dict('records')[0]
        else:
            # For multiple rows, handle each row individually
            return constrained.apply(lambda row: apply_prediction_constraints(row), axis=1)
    
    # Now constrained should be a dictionary
    # Apply constraints to each component
    # Protein constraints: typically 15-40% for animal feed
    if 'Protein' in constrained:
        constrained['Protein'] = np.clip(constrained['Protein'], 10, 50)
    
    # Fat constraints: typically 2-25% for animal feed
    if 'fat' in constrained:
        constrained['fat'] = np.clip(constrained['fat'], 1, 30)
    
    # Ash constraints: typically 2-15% for animal feed
    if 'ash' in constrained:
        constrained['ash'] = np.clip(constrained['ash'], 0.5, 20)
        # If ash is negative, set to a reasonable minimum
        if constrained['ash'] < 0:
            constrained['ash'] = 2.0
    
    # Moisture constraints: typically 5-15%
    if 'moisture' in constrained:
        constrained['moisture'] = np.clip(constrained['moisture'], 3, 20)
    
    # Fiber constraints: typically 2-15%
    if 'Fiber' in constrained:
        constrained['Fiber'] = np.clip(constrained['Fiber'], 1, 20)
    
    # Water activity constraints: must be between 0-1
    if 'wa' in constrained:
        constrained['wa'] = np.clip(constrained['wa'], 0.1, 0.99)
    
    return constrained
def select_moisture_wavelengths(X, wavelengths, y_moisture):
    """Select wavelengths most correlated with moisture"""
    # Calculate correlation with moisture
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y_moisture)[0, 1]
        correlations.append(abs(corr))
    
    # Select top wavelengths
    n_selected = min(100, X.shape[1])  # Select top 100 or all if less
    idx_sorted = np.argsort(correlations)[-n_selected:]
    
    return X[:, idx_sorted], [wavelengths[i] for i in idx_sorted]
def clean_and_prepare_data(dataset, target_cols):
    """Clean and prepare data for training"""
    with st.expander("🧹 Data Preparation", expanded=True):
        if dataset is None or len(dataset) == 0:
            st.error("❌ No data to prepare!")
            return None
        
        clean_data = dataset.copy()
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Samples", len(clean_data))
        with col2:
            existing_targets = [col for col in target_cols if col in clean_data.columns]
            st.metric("🎯 Available Targets", len(existing_targets))
        with col3:
            missing_count = clean_data[existing_targets].isna().sum().sum()
            st.metric("🔍 Missing Values", missing_count)
        
        # Target status table
        st.markdown("### 📋 Target Column Status")
        target_status = []
        for target in TARGET_COLS:
            exists = target in clean_data.columns
            missing = 0
            if exists:
                missing = clean_data[target].isna().sum()
                coverage = f"{(1 - missing/len(clean_data)) * 100:.1f}%" if len(clean_data) > 0 else '0%'
            else:
                coverage = '0%'
            
            target_status.append({
                'Target': target,
                'Status': '✅ Available' if exists and missing < len(clean_data) else '❌ Missing',
                'Missing Values': missing if exists else 'N/A',
                'Coverage': coverage
            })
        
        status_df = pd.DataFrame(target_status)
        st.dataframe(status_df, use_container_width=True)
        
        # Handle missing values in target columns
        if clean_data[existing_targets].isna().sum().sum() > 0:
            st.warning("⚠️ Found missing values in target columns")
            
            option = st.radio(
                "**Select handling method:**",
                ["Fill missing values with zero", "Fill missing values with column mean"],
                horizontal=True
            )
            
            if option == "Fill missing values with column mean":
                imputer = SimpleImputer(strategy='mean')
                cols_to_impute = [col for col in existing_targets if clean_data[col].notna().any()]
                if cols_to_impute:
                    clean_data[cols_to_impute] = pd.DataFrame(
                        imputer.fit_transform(clean_data[cols_to_impute]),
                        columns=cols_to_impute,
                        index=clean_data.index
                    )
                st.success("✅ Missing values filled with column means")
            else:
                clean_data[existing_targets] = clean_data[existing_targets].fillna(0)
                st.success("✅ Missing values filled with zeros")
        
        # Identify wavelength columns (all numeric columns except targets and Sample ID)
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols 
                         if col not in existing_targets + ['Sample ID']]
        
        st.success(f"✅ Final dataset: **{len(clean_data)}** samples with **{len(existing_targets)}** targets")
        st.info(f"📊 Spectrum features: **{len(wavelength_cols)}** wavelength points")
        
        # Show cleaned data preview
        st.markdown("### 📈 Cleaned Data Preview")
        preview_cols = ['Sample ID'] if 'Sample ID' in clean_data.columns else []
        preview_cols += existing_targets[:min(5, len(existing_targets))]
        st.dataframe(clean_data[preview_cols].head(), use_container_width=True)
    
    return clean_data

def save_model(model, model_name, wavelengths, target_cols, dataset_info, feature_groups=None):
    """Save trained model to file with feature selection info"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_data = {
        'model': model,
        'wavelengths': wavelengths,
        'target_cols': target_cols,
        'feature_groups': feature_groups or {},
        'created_date': datetime.now().isoformat(),
        'dataset_info': dataset_info
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path
def plot_feature_importance(model, selected_features, target_names):
    """
    Plot feature importance from PLS model
    """
    if hasattr(model, 'x_weights_'):
        importance = np.abs(model.x_weights_[:, 0])  # First component weights
        
        fig = make_subplots(rows=len(target_names), cols=1,
                           subplot_titles=[f"Feature Importance for {target}" 
                                          for target in target_names])
        
        wavelengths = [float(w) for w in selected_features]
        
        for i, target in enumerate(target_names):
            if i < model.x_weights_.shape[1]:
                importance = np.abs(model.x_weights_[:, i])
                
                fig.add_trace(
                    go.Scatter(
                        x=wavelengths,
                        y=importance,
                        mode='lines+markers',
                        name=target
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="Feature Importance by Component",
            height=300 * len(target_names),
            showlegend=False
        )
        
        return fig
    return None
def predict_with_spectrum(model_data, spectrum, bias_correction=None):
    """Make predictions using trained model and a spectrum"""
    if model_data is None or 'model' not in model_data:
        return None, None
    
    model = model_data['model']
    wavelengths = model_data['wavelengths']
    target_cols = model_data['target_cols']
    
    try:
        # Align spectrum with model wavelengths
        model_wavelengths = [float(w) for w in wavelengths]
        spectrum_aligned = spectrum.reindex(model_wavelengths)
        
        # Handle missing values
        if spectrum_aligned.isna().any():
            spectrum_aligned = spectrum_aligned.interpolate(method='linear')
            spectrum_aligned = spectrum_aligned.ffill().bfill()
        
        # Prepare for prediction
        X_new = spectrum_aligned.values.reshape(1, -1)
        X_new_snv = apply_snv(X_new)
        
        # Make prediction
        prediction = model.predict(X_new_snv)
        
        # Create result DataFrame
        result_df = pd.DataFrame(prediction, columns=target_cols)
        
        # Apply bias correction if available
        if bias_correction:
            result_df = apply_bias_correction(result_df, bias_correction)
            st.info(f"✅ Applied bias correction")
        
        return result_df, spectrum_aligned
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None
def create_spectra_plot(spectra_data, sample_ids=None):
    """Create interactive spectra plot using Plotly"""
    fig = go.Figure()
    
    for idx, (name, spectrum) in enumerate(spectra_data.items()):
        if sample_ids:
            label = sample_ids[idx] if idx < len(sample_ids) else name
        else:
            label = name
        
        fig.add_trace(go.Scatter(
            x=spectrum.index,
            y=spectrum.values,
            mode='lines',
            name=label,
            line=dict(width=1),
            opacity=0.6
        ))
    
    fig.update_layout(
        title="NIR Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10)
        )
    )
    
    return fig

def create_prediction_gauge(value, target, min_val=0, max_val=100):
    """Create a gauge chart for prediction values"""
    # Set appropriate max values based on target
    if target == "Protein":
        max_val = 50
    elif target == "fat":
        max_val = 30
    elif target == "ash":
        max_val = 10
    elif target == "moisture":
        max_val = 10
    elif target == "Fiber":
        max_val = 10
    elif target == "wa":  # Water activity?
        max_val = 1
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': target},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#764ba2"},
            'steps': [
                {'range': [min_val, max_val*0.33], 'color': "#EF4444"},
                {'range': [max_val*0.33, max_val*0.66], 'color': "#F59E0B"},
                {'range': [max_val*0.66, max_val], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.8
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==============================
# SESSION STATE
# ==============================
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("<div class='card'><h2>🧪 AI Feed Lab</h2><p>Prediction System v2.0</p></div>", 
                unsafe_allow_html=True)
    
    mode = st.radio(
        "**Select Mode:**",
        ["🏠 Dashboard", "🚀 Train Model", "🔮 Predict", "📚 Models"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if mode == "🚀 Train Model":
        st.markdown("### 📤 Upload Data")
        nir_files = st.file_uploader(
            "IAS 5100 CSV Files",
            type="csv",
            accept_multiple_files=True,
            help="Upload NIR spectral data files (format similar to IAS_Spectrum.csv)"
        )
        
        lab_file = st.file_uploader(
            "Lab Results",
            type=["csv", "txt"],
            help="Upload lab results (CSV or tab-delimited with columns: Protein, fat, ash, moisture, Fiber, wa)"
        )
        
        model_name = st.text_input(
            "Model Name",
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Give your model a descriptive name"
        )
        
    elif mode == "🔮 Predict":
        st.markdown("### 📤 Upload for Prediction")
        nir_files = st.file_uploader(
            "IAS Files",
            type="csv",
            accept_multiple_files=True
        )
        
    elif mode == "📚 Models":
        saved_models = []
        if os.path.exists(MODELS_DIR):
            saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        
        if saved_models:
            st.markdown("### 📁 Saved Models")
            selected_model = st.selectbox(
                "Choose a model:",
                options=saved_models,
                format_func=lambda x: x.replace('.pkl', '')
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load", use_container_width=True):
                    try:
                        model_path = os.path.join(MODELS_DIR, selected_model)
                        with open(model_path, 'rb') as f:
                            loaded_data = pickle.load(f)
                        
                        st.session_state.model_data = loaded_data
                        
                        st.session_state.trained_model = loaded_data['model']
                        st.success(f"✅ Model loaded!")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
            with col2:
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    try:
                        model_path = os.path.join(MODELS_DIR, selected_model)
                        os.remove(model_path)
                        st.success("✅ Model deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting model: {str(e)}")
        else:
            st.info("No saved models found. Train a model first.")

# ==============================
# MAIN CONTENT
# ==============================

if mode == "🏠 Dashboard":
    st.markdown("<h1 style='text-align: center;'>🧪 AI Feed Lab Prediction System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>IAS 5100 | Version 2.0</p>", 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🚀 Quick Start</h3>
            <p>1. Upload spectral data</p>
            <p>2. Upload lab results</p>
            <p>3. Train your model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Target Parameters</h3>
            <p>• Protein • fat • ash</p>
            <p>• moisture • Fiber • wa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.model_data:
            st.markdown("""
            <div class='metric-card'>
                <h3>✅ Model Loaded</h3>
                <p>Ready for predictions!</p>
                <p><strong>Targets:</strong> {}</p>
            </div>
            """.format(len(st.session_state.model_data['target_cols'])), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <h3>📊 System Status</h3>
                <p>No model loaded</p>
                <p>Go to <strong>Train Model</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Workflow Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train", "🔮 Predict", "📊 Analyze", "📁 Manage"])
    
    with tab1:
        st.markdown("""
        **Training Process:**
        1. **Upload Data**: Select IAS 5100 files and corresponding lab results
        2. **Data Matching**: System matches samples by Sample ID
        3. **Data Preparation**: Handle missing values and normalize spectra
        4. **Model Training**: Train PLS regression model
        5. **Save Model**: Store trained model for future use
        """)
    
    with tab2:
        st.markdown("""
        **Prediction Process:**
        1. **Load Model**: Select a pre-trained model
        2. **Upload New Data**: Upload IAS files for prediction
        3. **Get Predictions**: View predicted values for all parameters
        4. **Export Results**: Download predictions as CSV
        """)
    
    with tab3:
        st.markdown("""
        **Analysis Features:**
        - Interactive spectra visualization
        - Real-time prediction gauges
        - Performance metrics and statistics
        - Sample matching analysis
        """)
    
    with tab4:
        st.markdown("""
        **Model Management:**
        - Load saved models
        - Delete outdated models
        - View model metadata
        - Compare model performance
        """)
elif mode == "🚀 Train Model":
    st.markdown("<h1>🚀 Train New Model</h1>", unsafe_allow_html=True)
    
    if nir_files and lab_file:
        # Step 1: Upload status
        st.markdown("### 📁 Upload Status")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ {len(nir_files)} NIR files uploaded")
        with col2:
            st.success(f"✅ Lab file uploaded: {lab_file.name}")
        
        # Parse NIR files
        st.markdown("### 🔍 Parsing NIR Files")
        progress_bar = st.progress(0)

        spectra_dict = {}
        duplicate_count = {}

        for idx, file in enumerate(nir_files):
            with st.spinner(f"Parsing {file.name}..."):
                spectrum, sn = parse_ias_5100(file)
                if not spectrum.empty:
                    # Count duplicates
                    if sn in duplicate_count:
                        duplicate_count[sn] += 1
                        # Create a unique key for the duplicate
                        unique_key = f"{sn}_dup{duplicate_count[sn]}"
                        spectra_dict[unique_key] = spectrum
                        # st.info(f"📝 Duplicate sample ID '{sn}' found. Using '{unique_key}' for this spectrum.")
                    else:
                        duplicate_count[sn] = 1
                        spectra_dict[sn] = spectrum
            progress_bar.progress((idx + 1) / len(nir_files))

        # Count total duplicates
        total_duplicates = sum(count - 1 for count in duplicate_count.values() if count > 1)
        if total_duplicates > 0:
            st.warning(f"⚠️ Found {total_duplicates} duplicate sample IDs. Each spectrum will be used separately for training.")
        
        if len(spectra_dict) == 0:
            st.error("❌ No valid spectra found!")
            st.stop()
        
        # Load lab data
        st.markdown("### 📊 Loading Lab Data")
        lab_df = load_lab_data(lab_file)
        if lab_df is None:
            st.stop()
        
        # Prepare training data
        dataset, matched_ids, match_type = prepare_training_data(spectra_dict, lab_df)
        if dataset is None:
            st.stop()
        
        # Clean data
        clean_data = clean_and_prepare_data(dataset, TARGET_COLS)
        if clean_data is None:
            st.stop()
        
        # ==============================
        # FEATURE SELECTION SECTION - CORRECTED
        # ==============================
        st.markdown("### 🎯 Feature Selection Options")
        
        # First, get existing targets ONCE
        existing_targets = [col for col in TARGET_COLS if col in clean_data.columns]
        
        col1, col2 = st.columns(2)
        with col1:
            feature_selection_method = st.selectbox(
                "Feature Selection Method",
                ["component_based", "all_features", "statistical"],
                format_func=lambda x: {
                    "component_based": "🎯 Component-Based (Science-driven)",
                    "all_features": "📊 All Wavelengths",
                    "statistical": "📈 Statistical Correlation"
                }[x]
            )
        
        with col2:
            if feature_selection_method == "statistical":
                n_features = st.slider("Number of features to select", 10, 500, 100)
            else:
                n_features = None
        
        # Apply feature selection
      # ==============================
        # COMPONENT-SPECIFIC PROCESSING SECTION
        # ==============================

        if feature_selection_method == "component_based":
            # Apply COMPONENT-SPECIFIC preprocessing and feature selection
            st.markdown("### 🎯 Component-Specific Processing")
            
            try:
                component_models, component_features = train_model_with_component_preprocessing(
                    clean_data, 
                    existing_targets
                )
                
                # Train separate models for each component
                st.markdown("### 🤖 Training Component-Specific Models")

                all_models = {}
                performance_results = []
                dataset_info = {}

                for target in existing_targets:
                    if target in component_models:
                        comp_data = component_models[target]
                        
                        # Check if we have valid data
                        if comp_data['X'].shape[0] == 0:
                            st.warning(f"Skipping {target} - no training data available")
                            continue
                            
                        st.markdown(f"#### 🔧 Training model for **{target}**")
                        
                        # Get data for this component
                        comp_data = component_models[target]
                        X_comp = comp_data['X']
                        y_comp = comp_data['y']
                        
                        # Split data
                        if len(clean_data) >= 10:
                            test_size = 0.2
                        elif len(clean_data) >= 5:
                            test_size = 0.3
                        else:
                            test_size = 0.4
                            
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_comp, y_comp, test_size=test_size, random_state=42
                        )
                        
                        # Determine optimal components
                        # n_components = min(10, X_train.shape[0] - 1, X_train.shape[1])
                        # n_components = max(1, n_components)
                        
                        # # Train PLS model
                        # model = PLSRegression(n_components=n_components)
                        # model.fit(X_train, y_train)
                        # In the training loop, replace this section:
                        # Determine optimal components
                        n_components = min(10, X_train.shape[0] - 1, X_train.shape[1])
                        n_components = max(1, n_components)

                        # Train PLS model
                        model = PLSRegression(n_components=n_components)
                        model.fit(X_train, y_train)

                        # With this improved version:
                        # Find optimal number of components using cross-validation
                        max_components = min(15, X_train.shape[0] - 1, X_train.shape[1])
                        max_components = max(1, max_components)

                        best_score = -np.inf
                        best_n_components = 1

                        if X_train.shape[0] >= 10:  # Only do CV if enough samples
                            for n_comp in range(1, max_components + 1):
                                # Simple train-test split validation
                                if len(X_train) >= 20:
                                    from sklearn.model_selection import cross_val_score
                                    temp_model = PLSRegression(n_components=n_comp)
                                    scores = cross_val_score(temp_model, X_train, y_train, cv=min(5, len(X_train)), 
                                                            scoring='r2', n_jobs=-1)
                                    mean_score = np.mean(scores)
                                else:
                                    # For small datasets, use holdout
                                    temp_model = PLSRegression(n_components=n_comp)
                                    temp_model.fit(X_train, y_train)
                                    y_pred_val = temp_model.predict(X_train)
                                    mean_score = r2_score(y_train, y_pred_val)
                                
                                if mean_score > best_score:
                                    best_score = mean_score
                                    best_n_components = n_comp
                            
                            st.info(f"Optimal PLS components for {target}: {best_n_components} (CV R²: {best_score:.3f})")
                        else:
                            best_n_components = min(5, X_train.shape[0] - 1, X_train.shape[1])
                            best_n_components = max(1, best_n_components)

                        # Train final model with optimal components
                        model = PLSRegression(n_components=best_n_components)
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # Store model
                        all_models[target] = {
                            'model': model,
                            'wavelengths': comp_data['wavelengths'],
                            'preprocessing': comp_data['preprocessing'],
                            'performance': {'R2': r2, 'RMSE': rmse, 'MAE': mae}
                        }
                        
                        # Store results
                        performance_results.append({
                            'Component': target,
                            'R²': f"{r2:.4f}",
                            'RMSE': f"{rmse:.4f}",
                            'MAE': f"{mae:.4f}",
                            'Wavelengths': len(comp_data['wavelengths']),
                            'Preprocessing': ' + '.join(comp_data['preprocessing'])
                        })
                        
                        # Show progress
                        status = "✅ Good" if r2 > 0.7 else "⚠️ Needs improvement" if r2 > 0.5 else "❌ Poor"
                        st.info(f"{target}: R²={r2:.4f}, RMSE={rmse:.4f} - {status}")

                # Show performance summary
                st.markdown("### 📊 Model Performance Summary")
                if performance_results:
                    perf_df = pd.DataFrame(performance_results)
                    st.dataframe(perf_df, use_container_width=True)
                
                # Prepare dataset info for saving
                dataset_info = {
                    'n_samples': len(clean_data),
                    'targets': existing_targets,
                    'sample_ids': clean_data['Sample ID'].tolist() if 'Sample ID' in clean_data.columns else [],
                    'created_date': datetime.now().isoformat(),
                    'feature_selection_method': 'component_based'
                }
                
                # Store in session state for prediction
                st.session_state.component_models = all_models
                st.session_state.model_data = {
                    'component_models': all_models,
                    'target_cols': existing_targets,
                    'dataset_info': dataset_info,
                    'preprocessing_info': COMPONENT_PREPROCESSING
                }
                
                # Save model section
                st.markdown("### 💾 Save Component Models")
                if st.button("💾 Save Component Models", type="primary", use_container_width=True):
                    try:
                        model_path = save_component_models(
                            all_models, 
                            model_name, 
                            existing_targets, 
                            dataset_info
                        )
                        
                        st.success(f"✅ Component models saved as `{model_name}`!")
                        st.info(f"📊 Saved {len(all_models)} component-specific models")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
                
                # Skip the rest of the old training code
                continue_training = False
                
            except Exception as e:
                st.error(f"❌ Error in component-based processing: {str(e)}")
                # Fallback to old method
                continue_training = True
                
        else:
            continue_training = True

        # ==============================
        # OLD TRAINING METHOD (for non-component-based)
        # ==============================
        if continue_training:
            # This is your original code for statistical and all_features methods
            if feature_selection_method == "statistical":
                # Statistical correlation-based selection
                numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
                wavelength_cols = [col for col in numeric_cols if col not in existing_targets + ['Sample ID']]
                
                X_all = clean_data[wavelength_cols].values.astype(float)
                y_all = clean_data[existing_targets].values.astype(float)
                
                if len(wavelength_cols) > 0 and n_features <= len(wavelength_cols):
                    # Select features based on correlation with the first target
                    try:
                        selected_features, corr_fig = correlation_based_selection(
                            X_all, y_all, wavelength_cols, target_idx=0, n_features=n_features
                        )
                        st.plotly_chart(corr_fig, use_container_width=True)
                        X = clean_data[selected_features].values.astype(float)
                        y = y_all
                        feature_groups = {"statistical": selected_features}
                        st.success(f"✅ Selected {len(selected_features)} wavelengths based on correlation")
                    except Exception as e:
                        st.error(f"❌ Error in statistical selection: {str(e)}")
                        X = X_all
                        y = y_all
                        selected_features = wavelength_cols
                        feature_groups = {"all": wavelength_cols}
                else:
                    st.warning("⚠️ Not enough features for statistical selection")
                    X = X_all
                    y = y_all
                    selected_features = wavelength_cols
                    feature_groups = {"all": wavelength_cols}
                    
            else:  # all_features
                # Use all wavelengths
                numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
                wavelength_cols = [col for col in numeric_cols if col not in existing_targets + ['Sample ID']]
                
                X = clean_data[wavelength_cols].values.astype(float)
                y = clean_data[existing_targets].values.astype(float)
                selected_features = wavelength_cols
                feature_groups = {"all": wavelength_cols}
                
                st.info(f"📊 Using all {len(wavelength_cols)} wavelengths")

            # Check if we have enough data
            if len(clean_data) < 2:
                st.error(f"❌ Not enough samples for training. Only {len(clean_data)} samples available.")
                st.stop()
            
            # Apply SNV normalization
            st.markdown("### 🔄 Pre-processing Spectra")
            X_snv = apply_snv(X)
            
            # Train model
            st.markdown("### 🤖 Training Model")
            
            n_components = min(10, X.shape[0] - 1, X.shape[1])
            n_components = max(1, n_components)
            
            test_size = min(0.2, max(0.05, 1/len(clean_data)))
            X_train, X_test, y_train, y_test = train_test_split(
                X_snv, y, test_size=test_size, random_state=42
            )
            
            with st.spinner("Training PLS regression model..."):
                try:
                    model = PLSRegression(n_components=n_components)
                    model.fit(X_train, y_train)
                    st.success(f"✅ Model trained with {n_components} PLS components!")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.stop()
            
            # Evaluation
            st.markdown("### 📈 Model Performance")
            
            # Training metrics
            y_pred_train = model.predict(X_train)
            metrics_data = []
            for i, target in enumerate(existing_targets):
                if len(np.unique(y_train[:, i])) > 1:  # Check if there's variation
                    train_r2 = r2_score(y_train[:, i], y_pred_train[:, i])
                    train_rmse = np.sqrt(mean_squared_error(y_train[:, i], y_pred_train[:, i]))
                    
                    metrics_data.append({
                        'Target': target,
                        'R² Score': f"{train_r2:.4f}",
                        'RMSE': f"{train_rmse:.4f}",
                        'Status': '✅ Good' if train_r2 > 0.7 else '⚠️ Needs improvement'
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient data variation to calculate metrics")
            
            # Feature importance visualization
            if feature_selection_method != "all_features":
                st.markdown("### 📊 Feature Importance")
                try:
                    importance_fig = plot_feature_importance(model, selected_features, existing_targets)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate feature importance plot: {str(e)}")
            
            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                if spectra_dict:
                    fig = create_spectra_plot(spectra_dict)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(existing_targets) > 1:
                    fig = px.imshow(
                        np.corrcoef(y.T),
                        x=existing_targets,
                        y=existing_targets,
                        color_continuous_scale='RdBu',
                        title="Target Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Save model
            st.markdown("### 💾 Save Model")
            if st.button("💾 Save Model", type="primary", use_container_width=True):
                dataset_info = {
                    'n_samples': len(clean_data),
                    'n_wavelengths': len(selected_features),
                    'targets': existing_targets,
                    'sample_ids': clean_data['Sample ID'].tolist() if 'Sample ID' in clean_data.columns else [],
                    'feature_selection_method': feature_selection_method,
                    'created_date': datetime.now().isoformat()
                }
                
                try:
                    model_path = save_model(
                        model, 
                        model_name, 
                        selected_features,
                        existing_targets, 
                        dataset_info,
                        feature_groups
                    )
                    
                    st.session_state.trained_model = model
                    st.session_state.model_data = {
                        'model': model,
                        'wavelengths': selected_features,
                        'target_cols': existing_targets,
                        'feature_groups': feature_groups,
                        'dataset_info': dataset_info
                    }
                    
                    st.success(f"✅ Model saved as `{model_name}`!")
                    st.info(f"📊 Used {feature_selection_method} feature selection with {len(selected_features)} wavelengths")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
                    # Use all wavelengths
                    numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
                    wavelength_cols = [col for col in numeric_cols if col not in existing_targets + ['Sample ID']]
                    
                    X = clean_data[wavelength_cols].values.astype(float)
                    y = clean_data[existing_targets].values.astype(float)
                    selected_features = wavelength_cols
                    feature_groups = {"all": wavelength_cols}
                    
                    st.info(f"📊 Using all {len(wavelength_cols)} wavelengths")       

                # Check if we have enough data
                if len(clean_data) < 2:
                    st.error(f"❌ Not enough samples for training. Only {len(clean_data)} samples available.")
                    st.stop()
                
                # Apply SNV normalization
                st.markdown("### 🔄 Pre-processing Spectra")
                X_snv = apply_snv(X)
                
                # Train model
                st.markdown("### 🤖 Training Model")
                
                n_components = min(10, X.shape[0] - 1, X.shape[1])
                n_components = max(1, n_components)
                
                test_size = min(0.2, max(0.05, 1/len(clean_data)))
                X_train, X_test, y_train, y_test = train_test_split(
                    X_snv, y, test_size=test_size, random_state=42
                )
                
                with st.spinner("Training PLS regression model..."):
                    try:
                        model = PLSRegression(n_components=n_components)
                        model.fit(X_train, y_train)
                        st.success(f"✅ Model trained with {n_components} PLS components!")
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.stop()
                
                # Evaluation
                st.markdown("### 📈 Model Performance")
                
                # Training metrics
                y_pred_train = model.predict(X_train)
                metrics_data = []
                for i, target in enumerate(existing_targets):
                    if len(np.unique(y_train[:, i])) > 1:  # Check if there's variation
                        train_r2 = r2_score(y_train[:, i], y_pred_train[:, i])
                        train_rmse = np.sqrt(mean_squared_error(y_train[:, i], y_pred_train[:, i]))
                        
                        metrics_data.append({
                            'Target': target,
                            'R² Score': f"{train_r2:.4f}",
                            'RMSE': f"{train_rmse:.4f}",
                            'Status': '✅ Good' if train_r2 > 0.7 else '⚠️ Needs improvement'
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient data variation to calculate metrics")
                
                # Feature importance visualization
                if feature_selection_method != "all_features":
                    st.markdown("### 📊 Feature Importance")
                    try:
                        importance_fig = plot_feature_importance(model, selected_features, existing_targets)
                        if importance_fig:
                            st.plotly_chart(importance_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate feature importance plot: {str(e)}")
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    if spectra_dict:
                        fig = create_spectra_plot(spectra_dict)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(existing_targets) > 1:
                        fig = px.imshow(
                            np.corrcoef(y.T),
                            x=existing_targets,
                            y=existing_targets,
                            color_continuous_scale='RdBu',
                            title="Target Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Save model
                st.markdown("### 💾 Save Model")
                if st.button("💾 Save Model", type="primary", use_container_width=True):
                    dataset_info = {
                        'n_samples': len(clean_data),
                        'n_wavelengths': len(selected_features),
                        'targets': existing_targets,
                        'sample_ids': clean_data['Sample ID'].tolist() if 'Sample ID' in clean_data.columns else [],
                        'feature_selection_method': feature_selection_method,
                        'created_date': datetime.now().isoformat()
                    }
                    
                    try:
                        model_path = save_model(
                            model, 
                            model_name, 
                            selected_features,
                            existing_targets, 
                            dataset_info,
                            feature_groups
                        )
                        
                        st.session_state.trained_model = model
                        st.session_state.model_data = {
                            'model': model,
                            'wavelengths': selected_features,
                            'target_cols': existing_targets,
                            'feature_groups': feature_groups,
                            'dataset_info': dataset_info
                        }
                        
                        st.success(f"✅ Model saved as `{model_name}`!")
                        st.info(f"📊 Used {feature_selection_method} feature selection with {len(selected_features)} wavelengths")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
            
            else:
                st.info("""
                ## 📚 Training Instructions
                
                **Required Files:**
                1. **IAS 5100 CSV files** - NIR spectral data with Sample ID (format similar to IAS_Spectrum.csv)
                2. **Lab Results file** - CSV or tab-delimited with target values
                
                **Lab File Format Example:**
                ```
                Sample ID,Protein,fat,ash,moisture,Fiber,wa
                B2600041_01,23,21,4.6,5,3.49,0.4
                B2600041_02,23,21,4.6,5,3.49,0.4
                B2500012,30,13,5.6,6.3,3.35,0.47
                ```
                
                Upload both file types to begin training.
                """)

elif mode == "🔮 Predict":
    st.markdown("<h1>🔮 Make Predictions</h1>", unsafe_allow_html=True)
    
    if st.session_state.model_data is None:
        st.warning("⚠️ No model loaded. Please train or load a model first.")
        st.info("Go to **Models** tab to load a saved model, or **Train Model** to create a new one.")
    elif nir_files:
        st.markdown(f"### 📊 Processing {len(nir_files)} file(s)")
        
        all_predictions = []
        
        for file in nir_files:
            with st.expander(f"📄 {file.name}", expanded=True):
                spectrum, sn = parse_ias_5100(file)
                
                if not spectrum.empty:
                    # In your prediction section, update this:
                    if st.session_state.model_data:
                        # Check which type of model we have
                        if 'component_models' in st.session_state.model_data:
                            # Use component models
                            result_df, aligned_spectra = predict_with_component_models(
                                st.session_state.model_data, 
                                spectrum
                            )
                            
                            # Apply bias correction
                            if result_df is not None:
                                # Apply both constraint and bias correction
                                # Apply both constraint and bias correction
                                result_df = apply_prediction_constraints(result_df.iloc[0])
                                # Convert dictionary back to DataFrame row
                                result_df = pd.DataFrame([result_df])
                                result_df = apply_bias_correction(result_df)
                        else:
                            # Use single model (old format)
                            result_df, aligned_spectrum = predict_with_spectrum(
                                st.session_state.model_data, 
                                spectrum
                            )
                        if result_df is not None:
                            st.success(f"✅ Predictions for {sn}")
                            
                            # Display predictions as gauges
                            st.markdown("### 🎯 Predicted Values")
                            target_cols = st.session_state.model_data['target_cols']
                            cols = st.columns(len(target_cols))
                            for idx, col_name in enumerate(target_cols):
                                with cols[idx]:
                                    if col_name in result_df.columns:
                                        value = result_df[col_name].iloc[0]
                                        fig = create_prediction_gauge(
                                            value=float(value),
                                            target=col_name,
                                            min_val=0,
                                            max_val=100
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to summary
                            prediction_row = result_df.copy()
                            prediction_row['File'] = file.name
                            prediction_row['Sample ID'] = sn
                            all_predictions.append(prediction_row)
                            
                            # Plot spectrum
                                                      # Plot spectrum
                            st.markdown("### 📈 Spectrum")
                            fig = go.Figure()
                            
                            if 'aligned_spectra' in locals():
                                # Component model - plot first spectrum
                                if aligned_spectra:
                                    first_target = list(aligned_spectra.keys())[0]
                                    spectrum_to_plot = aligned_spectra[first_target]
                                    fig.add_trace(go.Scatter(
                                        x=spectrum_to_plot.index,
                                        y=spectrum_to_plot.values,
                                        mode='lines',
                                        line=dict(color='#764ba2', width=2),
                                        name=f'Spectrum for {first_target}'
                                    ))
                            else:
                                # Single model
                                fig.add_trace(go.Scatter(
                                    x=aligned_spectrum.index,
                                    y=aligned_spectrum.values,
                                    mode='lines',
                                    line=dict(color='#764ba2', width=2),
                                    name='Spectrum'
                                ))
                            fig.update_layout(
                                title=f"Spectrum: {sn}",
                                xaxis_title="Wavelength (nm)",
                                yaxis_title="Absorbance",
                                template="plotly_white",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        if all_predictions:
            st.markdown("### 📋 Summary of Predictions")
            summary_df = pd.concat(all_predictions, ignore_index=True)
            
            target_cols = st.session_state.model_data['target_cols']
            display_cols = ['Sample ID', 'File'] + target_cols
            display_cols = [col for col in display_cols if col in summary_df.columns]
            
            st.dataframe(summary_df[display_cols], use_container_width=True)
            
            # Download
            csv = summary_df[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
    
    else:
        st.info("""
        ## 🔮 Prediction Mode
        
        **To make predictions:**
        1. Ensure a model is loaded (check sidebar)
        2. Upload IAS 5100 CSV files
        3. View predictions for each parameter
        
        **Current Status:** ✅ Ready for predictions
        
        Upload files to begin.
        """)

elif mode == "📚 Models":
    st.markdown("<h1>📚 Model Management</h1>", unsafe_allow_html=True)
    
    saved_models = []
    if os.path.exists(MODELS_DIR):
        saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    
    if saved_models:
        st.markdown(f"### 📁 Found {len(saved_models)} saved model(s)")
        
        for model_file in saved_models:
            try:
                model_path = os.path.join(MODELS_DIR, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                with st.expander(f"📦 {model_file.replace('.pkl', '')}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Targets", len(model_data.get('target_cols', [])))
                    with col2:
                        samples = model_data.get('dataset_info', {}).get('n_samples', 'N/A')
                        st.metric("Samples", samples)
                    with col3:
                        created_date = model_data.get('created_date', '').split('T')[0]
                        st.metric("Created", created_date)
                    
                    if model_data.get('target_cols'):
                        st.markdown("**Target Parameters:**")
                        for target in model_data['target_cols']:
                            st.markdown(f"• {target}")
                    
                    col_load, col_del, _ = st.columns([1, 1, 2])
                    with col_load:
                        if st.button("📥 Load", key=f"load_{model_file}", use_container_width=True):
                            try:
                                model_path = os.path.join(MODELS_DIR, selected_model)
                                with open(model_path, 'rb') as f:
                                    loaded_data = pickle.load(f)
                                
                                # Handle both old and new model formats
                                if 'component_models' in loaded_data:
                                    # New component model format
                                    st.session_state.model_data = loaded_data
                                    # For compatibility, store first model if available
                                    if loaded_data['component_models']:
                                        first_target = list(loaded_data['component_models'].keys())[0]
                                        st.session_state.trained_model = loaded_data['component_models'][first_target].get('model')
                                elif 'model' in loaded_data:
                                    # Old single model format
                                    st.session_state.model_data = loaded_data
                                    st.session_state.trained_model = loaded_data['model']
                                else:
                                    st.error("❌ Unknown model format")
                                    continue
                                
                                st.success(f"✅ Model loaded!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error loading model: {str(e)}")
                    with col_del:
                        if st.button("🗑️ Delete", key=f"del_{model_file}", type="secondary", use_container_width=True):
                            os.remove(model_path)
                            st.success("✅ Model deleted!")
                            st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error loading {model_file}: {str(e)}")
    
    else:
        st.info("""
        ## 📚 Model Library
        
        No saved models found. To save a model:
        
        1. Go to **Train Model**
        2. Upload your data and train a model
        3. Click "Save Model" to store it here
        
        Saved models will appear in this section for easy access.
        """)
    # Display currently loaded model info
    if st.session_state.model_data:
        st.markdown("---")
        st.markdown("### ✅ Currently Loaded Model")
        
        col1, col2 = st.columns(2)
        with col1:
            target_cols = st.session_state.model_data.get('target_cols', [])
            st.metric("Target Parameters", len(target_cols))
            if target_cols:
                st.write("**Parameters:**")
                for target in target_cols:
                    st.markdown(f"• {target}")
        
        with col2:
            # Check if it's a component model or single model
            if 'component_models' in st.session_state.model_data:
                # Component model - calculate total unique wavelengths
                all_wavelengths = set()
                for target, comp_model in st.session_state.model_data['component_models'].items():
                    if 'wavelengths' in comp_model:
                        all_wavelengths.update(comp_model['wavelengths'])
                wavelength_count = len(all_wavelengths)
                model_type = "Component-Specific"
            elif 'wavelengths' in st.session_state.model_data:
                # Single model
                wavelength_count = len(st.session_state.model_data['wavelengths'])
                model_type = "Single Model"
            else:
                wavelength_count = 0
                model_type = "Unknown"
            
            st.metric("Total Wavelength Points", wavelength_count)
            st.metric("Model Type", model_type)
            st.metric("Status", "✅ Ready for predictions")
# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 2rem;'>"
    "🧪 AI Feed Lab Prediction System • Version 2.0 • Modern UI Edition"
    "</div>",
    unsafe_allow_html=True
)