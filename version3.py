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

# Suppress warnings py -m streamlit run app.py

warnings.filterwarnings('ignore')
# Add this near the top of your code with other constants
# Update your COMPONENT_PREPROCESSING dictionary
# Updated Preprocessing Sequences
COMPONENT_PREPROCESSING = {
    'protein': ['snv'],
   'fat': ['msc', 'derivative', 'snv'],  # Added MSC and derivative
    'ash': ['savgol', 'snv'],  # Added derivative (savgol) for better baseline correction
    'moisture': ['msc', 'savgol', 'snv'], # Complete sequence for moisture
    'fiber': ['snv'],
    'wa': ['snv']
}

# Updated Wavelength Selection (matching specific absorption bands)
COMPONENT_WAVELENGTH_RANGES = {
    'protein': [(1150, 1300), (1480, 1650)],
    'fat': [(1190, 1210), (1700, 1740)],  # C-H stretching region
    # 'fat': [(1190, 1210), (1700, 1740)], # Targeted C-H stretching
    'ash': [(900, 1700)],                # All wavelengths (correlation based)
    'moisture': [(920, 1020), (1350, 1550)],
    'fiber': [(1050, 1400)],
    'wa': [(900, 1700)]
}
# COMPONENT_WAVELENGTHS = {
#     'Moisture': [(950, 1000), (1400, 1500)],  # Primary water absorption bands
#     'Protein': [(1180, 1250), (1500, 1600)],   # N-H peptide bonds
    # 'Fat': [(1190, 1210), (1700, 1740)],       # C-H stretching (lipids/oils)
#     'Fiber': [(1100, 1350)],                   # Complex carbohydrates
#     # Ash: statistical correlation only
# }

# Create a mapping of wavelength regions to components
# WAVELENGTH_REGIONS = {
#     'Moisture_1': (970, 1000),      # Primary O-H absorption
#     'Moisture_2': (1450, 1500),     # Secondary O-H absorption
#     'Protein_1': (1180, 1250),      # N-H peptide bonds
#     'Protein_2': (1500, 1600),      # N-H overtones
#     'Fat_1': (1190, 1210),          # C-H stretching
#     'Fat_2': (1720, 1740),          # C-H overtone
#     'Fiber_region': (1100, 1350),   # Carbohydrate structures
# }

# # Add this near the top of your code with other constants
# # Update the preprocessing dictionary
# # Update with better preprocessing strategies
# COMPONENT_PREPROCESSING = {
#     'moisture': ['savgol', 'snv'],
#     'Protein': ['msc', 'derivative'],  # Try MSC + derivative for protein
#     'fat': ['snv'],  # MSC then SNV for fat
#     'ash': ['snv', 'msc'],  # SNV first then MSC for ash
#     'Fiber': ['msc'],  # Just MSC for fiber
#     'wa': ['snv']
# }

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
# ==============================
# ENHANCED FEATURE SELECTION FUNCTIONS
# ==============================
def create_correlation_analysis(X, y, wavelength_cols, target_names):
    """Create correlation analysis between features and targets"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Calculate correlations
    correlation_matrix = np.zeros((len(wavelength_cols), len(target_names)))
    
    for i in range(len(wavelength_cols)):
        for j in range(len(target_names)):
            corr = np.corrcoef(X[:, i], y[:, j])[0, 1]
            correlation_matrix[i, j] = corr
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.T,
        x=[float(w) for w in wavelength_cols],
        y=target_names,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature-Target Correlation Matrix",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Target Variables",
        height=400
    )
    
    return fig, correlation_matrix

def select_features_by_correlation(X, y, wavelength_cols, target_names, method='highest_abs', n_features=100):
    """Select features based on correlation with targets"""
    # Calculate maximum absolute correlation for each feature across all targets
    max_correlations = np.zeros(len(wavelength_cols))
    
    for i in range(len(wavelength_cols)):
        correlations = []
        for j in range(len(target_names)):
            if len(np.unique(y[:, j])) > 1:  # Check if target has variation
                corr = np.corrcoef(X[:, i], y[:, j])[0, 1]
                correlations.append(abs(corr))
        max_correlations[i] = np.max(correlations) if correlations else 0
    
    # Select features based on method
    if method == 'highest_abs':
        # Select features with highest maximum absolute correlation
        top_idx = np.argsort(max_correlations)[-n_features:]
    elif method == 'above_threshold':
        # Select features above correlation threshold
        threshold = np.percentile(max_correlations[max_correlations > 0], 75)
        top_idx = np.where(max_correlations >= threshold)[0]
        if len(top_idx) > n_features:
            top_idx = top_idx[np.argsort(max_correlations[top_idx])[-n_features:]]
    elif method == 'balanced':
        # Select balanced number of features from different correlation levels
        sorted_idx = np.argsort(max_correlations)
        n_per_level = n_features // 3
        top_idx = np.concatenate([
            sorted_idx[-n_per_level:],  # Highest correlations
            sorted_idx[-(2*n_per_level):-n_per_level],  # Medium correlations
            sorted_idx[:n_per_level]  # Lowest correlations
        ])
    
    selected_features = [wavelength_cols[i] for i in top_idx]
    
    return selected_features, max_correlations, top_idx

def feature_selection_wizard(clean_data, existing_targets):
    """Interactive feature selection wizard"""
    with st.expander("🔍 Feature Selection Wizard", expanded=True):
        st.markdown("### 🎯 Intelligent Feature Selection")
        
        # Get wavelength columns
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols if col not in existing_targets + ['Sample ID']]
        
        if len(wavelength_cols) == 0:
            st.error("No wavelength columns found!")
            return None, None, None
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(wavelength_cols))
        with col2:
            st.metric("Targets", len(existing_targets))
        with col3:
            st.metric("Samples", len(clean_data))
        
        # Step 1: Correlation Analysis
        st.markdown("#### 📊 Step 1: Correlation Analysis")
        
        X_all = clean_data[wavelength_cols].values.astype(float)
        y_all = clean_data[existing_targets].values.astype(float)
        
        # Create correlation heatmap
        corr_fig, correlation_matrix = create_correlation_analysis(
            X_all, y_all, wavelength_cols, existing_targets
        )
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Step 2: Selection Method
        st.markdown("#### 🎯 Step 2: Selection Method")
        
        col1, col2 = st.columns(2)
        with col1:
            selection_method = st.selectbox(
                "Selection Algorithm",
                [
                    "component_based", 
                    "correlation_based", 
                    "hybrid", 
                    "pca_reduction",
                    "all_features"
                ],
                format_func=lambda x: {
                    "component_based": "🧪 Component-Based (Science)",
                    "correlation_based": "📈 Correlation-Based",
                    "hybrid": "🤝 Hybrid Approach",
                    "pca_reduction": "🔢 PCA Feature Reduction",
                    "all_features": "📊 All Features"
                }[x]
            )
        
        with col2:
            if selection_method != "all_features":
                n_features = st.slider(
                    "Number of features to select",
                    min_value=10,
                    max_value=min(500, len(wavelength_cols)),
                    value=min(100, len(wavelength_cols)),
                    help="Select optimal number of features (10-500)"
                )
            else:
                n_features = len(wavelength_cols)
        
        # Step 3: Algorithm-specific settings
        if selection_method == "correlation_based":
            st.markdown("#### ⚙️ Correlation Settings")
            corr_method = st.radio(
                "Correlation method:",
                ["highest_abs", "above_threshold", "balanced"],
                format_func=lambda x: {
                    "highest_abs": "Highest Absolute Correlation",
                    "above_threshold": "Above Correlation Threshold",
                    "balanced": "Balanced Selection"
                }[x],
                horizontal=True
            )
        elif selection_method == "hybrid":
            st.markdown("#### ⚙️ Hybrid Settings")
            col1, col2 = st.columns(2)
            with col1:
                component_weight = st.slider("Component knowledge weight", 0.0, 1.0, 0.5)
            with col2:
                correlation_weight = st.slider("Correlation weight", 0.0, 1.0, 0.5)
        
        # Step 4: Apply Selection
        st.markdown("#### 🚀 Step 3: Apply Feature Selection")
        
        if st.button("🔍 Analyze and Select Features", type="primary"):
            with st.spinner("Analyzing features..."):
                
                if selection_method == "component_based":
                    # Component-based selection
                    selected_features, feature_groups = select_informative_wavelengths(
                        wavelength_cols, method='component_based'
                    )
                    
                    # If we have too many features, select top N
                    if len(selected_features) > n_features:
                        # Calculate correlation scores to select best within component regions
                        X_comp = clean_data[selected_features].values
                        max_correlations = np.zeros(len(selected_features))
                        
                        for i in range(len(selected_features)):
                            correlations = []
                            for j in range(len(existing_targets)):
                                if len(np.unique(y_all[:, j])) > 1:
                                    corr = np.corrcoef(X_comp[:, i], y_all[:, j])[0, 1]
                                    correlations.append(abs(corr))
                            max_correlations[i] = np.max(correlations) if correlations else 0
                        
                        top_idx = np.argsort(max_correlations)[-n_features:]
                        final_features = [selected_features[i] for i in top_idx]
                    else:
                        final_features = selected_features
                    
                    st.success(f"✅ Selected {len(final_features)} features from component regions")
                
                elif selection_method == "correlation_based":
                    # Correlation-based selection
                    final_features, max_correlations, top_idx = select_features_by_correlation(
                        X_all, y_all, wavelength_cols, existing_targets,
                        method=corr_method, n_features=n_features
                    )
                    
                    # Plot correlation distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=max_correlations,
                        name='All Features',
                        marker_color='lightgray',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Histogram(
                        x=max_correlations[top_idx],
                        name='Selected Features',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title="Correlation Distribution",
                        xaxis_title="Maximum Absolute Correlation",
                        yaxis_title="Count",
                        barmode='overlay',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Selected {len(final_features)} features based on correlation")
                
                elif selection_method == "hybrid":
                    # Hybrid approach: combine component knowledge and correlation
                    
                    # 1. Get component-based features
                    comp_features, _ = select_informative_wavelengths(
                        wavelength_cols, method='component_based'
                    )
                    
                    # 2. Get correlation-based features
                    corr_features, max_correlations, _ = select_features_by_correlation(
                        X_all, y_all, wavelength_cols, existing_targets,
                        method='highest_abs', n_features=n_features
                    )
                    
                    # 3. Combine with weights
                    all_features = list(set(comp_features + corr_features))
                    
                    if len(all_features) > n_features:
                        # Score features based on hybrid approach
                        feature_scores = {}
                        for feature in all_features:
                            idx = wavelength_cols.index(feature)
                            score = 0
                            
                            # Component score
                            if feature in comp_features:
                                score += component_weight
                            
                            # Correlation score
                            if feature in corr_features:
                                corr_idx = corr_features.index(feature)
                                score += correlation_weight * max_correlations[idx]
                            
                            feature_scores[feature] = score
                        
                        # Select top N features
                        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                        final_features = [f[0] for f in sorted_features[:n_features]]
                    else:
                        final_features = all_features
                    
                    st.success(f"✅ Selected {len(final_features)} features using hybrid approach")
                
                elif selection_method == "pca_reduction":
                    # PCA-based feature reduction
                    from sklearn.decomposition import PCA
                    
                    # First, reduce dimensions with PCA
                    pca = PCA(n_components=min(50, X_all.shape[1]))
                    X_pca = pca.fit_transform(X_all)
                    
                    # Get most important original features
                    pca_components = np.abs(pca.components_)
                    feature_importance = np.sum(pca_components[:10, :], axis=0)  # Sum first 10 PCs
                    
                    # Select top features
                    top_idx = np.argsort(feature_importance)[-n_features:]
                    final_features = [wavelength_cols[i] for i in top_idx]
                    
                    # Plot PCA variance
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                        y=pca.explained_variance_ratio_,
                        name='Explained Variance'
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                        y=np.cumsum(pca.explained_variance_ratio_),
                        name='Cumulative Variance',
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title="PCA Variance Explained",
                        xaxis_title="Principal Component",
                        yaxis_title="Explained Variance Ratio",
                        yaxis2=dict(
                            title="Cumulative Variance",
                            overlaying='y',
                            side='right'
                        ),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Selected {len(final_features)} features using PCA")
                
                else:  # all_features
                    final_features = wavelength_cols
                    st.info(f"📊 Using all {len(final_features)} features")
                
                # Display selected features summary
                st.markdown("#### 📋 Selection Summary")
                
                # Calculate and display statistics
                X_selected = clean_data[final_features].values.astype(float)
                
                # Calculate feature statistics
                feature_means = np.mean(X_selected, axis=0)
                feature_stds = np.std(X_selected, axis=0)
                
                # Create summary table
                summary_data = []
                for i, feature in enumerate(final_features[:20]):  # Show first 20
                    wl = float(feature)
                    summary_data.append({
                        'Wavelength': f"{wl:.1f} nm",
                        'Mean': f"{feature_means[i]:.4f}",
                        'Std': f"{feature_stds[i]:.4f}",
                        'Status': '✅ Selected'
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                if len(final_features) > 20:
                    st.info(f"... and {len(final_features) - 20} more features")
                
                # Calculate reduction ratio
                reduction_ratio = 1 - (len(final_features) / len(wavelength_cols))
                st.metric(
                    "Feature Reduction", 
                    f"{reduction_ratio*100:.1f}%",
                    f"From {len(wavelength_cols)} to {len(final_features)}"
                )
                
                # Prepare data for training
                X = clean_data[final_features].values.astype(float)
                y = clean_data[existing_targets].values.astype(float)
                
                return X, y, final_features, selection_method
        
        return None, None, None, None

def plot_feature_correlation_network(selected_features, correlation_matrix, threshold=0.7):
    """Plot feature correlation network"""
    import networkx as nx
    
    if len(selected_features) > 50:  # Limit for visualization
        st.warning("Too many features for network visualization (max 50)")
        return None
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    wavelengths = [float(w) for w in selected_features]
    for i, wl in enumerate(wavelengths):
        G.add_node(i, wavelength=wl, label=f"{wl:.1f}nm")
    
    # Add edges based on correlation
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=correlation_matrix[i, j])
    
    # Create Plotly network visualization
    pos = nx.spring_layout(G)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Wavelength: {wavelengths[node]:.1f} nm")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            color='blue',
            line=dict(width=2)
        )
    )
    
    fig = go.Figure(data=edge_trace + [node_trace],
                   layout=go.Layout(
                       title='Feature Correlation Network',
                       showlegend=False,
                       hovermode='closest',
                       height=400
                   ))
    
    return fig

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

def apply_bias_correction(predictions):
    """Apply bias correction based on known prediction errors"""
    corrected = predictions.copy()
    
    # Ensure it's a DataFrame
    if not isinstance(corrected, pd.DataFrame):
        corrected = pd.DataFrame([corrected])
    
    # FIX: Enhanced fat correction
    if 'fat' in corrected.columns:
        current_fat = corrected['fat'].iloc[0]
        
        # If fat is too high (>25), apply stronger correction
        if current_fat > 25:
            corrected['fat'] = corrected['fat'] * 0.6  # Reduce by 40%
        elif current_fat > 20:
            corrected['fat'] = corrected['fat'] * 0.7  # Reduce by 30%
        elif current_fat > 15:
            corrected['fat'] = corrected['fat'] * 0.8  # Reduce by 20%
        else:
            corrected['fat'] = corrected['fat'] * 0.9  # Reduce by 10%
        
        # Keep within realistic bounds
        corrected['fat'] = np.clip(corrected['fat'], 5, 20)
        
        # If still unrealistic, use correlation with protein
        if corrected['fat'].iloc[0] > 18 and 'Protein' in corrected.columns:
            # Fat is typically 50-70% of protein in animal feed
            corrected['fat'] = corrected['Protein'] * 0.6
    
    # Protein correction
    if 'Protein' in corrected.columns:
        corrected['Protein'] = corrected['Protein'] * 0.95
        corrected['Protein'] = np.clip(corrected['Protein'], 15, 40)
    
    # Ash correction - should be positive and reasonable
    if 'ash' in corrected.columns:
        if corrected['ash'].iloc[0] < 0 or corrected['ash'].iloc[0] > 15:
            if 'Protein' in corrected.columns:
                corrected['ash'] = corrected['Protein'] * 0.22  # Ash ~22% of protein
            else:
                corrected['ash'] = 5.5
        corrected['ash'] = np.clip(corrected['ash'], 2, 12)
    
    # Other components...
    if 'moisture' in corrected.columns:
        corrected['moisture'] = np.clip(corrected['moisture'] * 0.92, 3, 15)
    
    if 'Fiber' in corrected.columns:
        corrected['Fiber'] = np.clip(corrected['Fiber'] * 1.1, 2, 10)
    
    if 'wa' in corrected.columns:
        corrected['wa'] = np.clip(corrected['wa'] * 0.85, 0.2, 0.8)
    
    return corrected
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

def save_model(model, model_name, wavelengths, target_cols, dataset_info):
    """Save trained model to file"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_data = {
        'model': model,
        'wavelengths': wavelengths,
        'target_cols': target_cols,
        'created_date': datetime.now().isoformat(),
        'dataset_info': dataset_info
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path

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
                        
                        # Handle both model formats
                        if 'component_models' in loaded_data:
                            # Component-based model format
                            st.session_state.model_data = loaded_data
                            # Set trained_model to first component model (for compatibility)
                            if loaded_data['component_models']:
                                first_target = list(loaded_data['component_models'].keys())[0]
                                st.session_state.trained_model = loaded_data['component_models'][first_target].get('model')
                        elif 'model' in loaded_data:
                            # Old single model format
                            st.session_state.model_data = loaded_data
                            st.session_state.trained_model = loaded_data['model']
                        else:
                            st.error("❌ Unknown model format")
                        
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
                    # Clean the sample ID - remove any whitespace
                    sn = str(sn).strip()
                    
                    # Count duplicates
                    if sn in duplicate_count:
                        duplicate_count[sn] += 1
                        # Create a unique key for the duplicate
                        unique_key = f"{sn}_dup{duplicate_count[sn]}"
                        spectra_dict[unique_key] = spectrum
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
        
        # Clean lab data Sample IDs
        if 'Sample ID' in lab_df.columns:
            lab_df['Sample ID'] = lab_df['Sample ID'].astype(str).str.strip()
        
        # Show data for debugging
        st.markdown("### 🔍 Data Matching Debug")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Spectra Sample IDs:**")
            for sid in list(spectra_dict.keys())[:10]:
                st.write(f"- {sid}")
        with col2:
            st.write("**Lab Sample IDs:**")
            if 'Sample ID' in lab_df.columns:
                for sid in lab_df['Sample ID'].head(10).tolist():
                    st.write(f"- {sid}")
        
        # Try to find matches with different patterns
        st.markdown("### 🤝 Matching Strategies")
        
        # Strategy 1: Exact match
        exact_matches = []
        for spec_id in spectra_dict.keys():
            clean_spec_id = str(spec_id).strip()
            lab_match = lab_df[lab_df['Sample ID'] == clean_spec_id]
            if not lab_match.empty:
                exact_matches.append(clean_spec_id)
        
        if exact_matches:
            st.success(f"✅ Found {len(exact_matches)} exact matches")
        else:
            st.warning("⚠️ No exact matches found. Trying partial matching...")
            
            # Strategy 2: Partial match (e.g., B2500343 might match B2500xxx)
            partial_matches = []
            for spec_id in spectra_dict.keys():
                clean_spec_id = str(spec_id).strip()
                # Try matching first 5 characters
                prefix = clean_spec_id[:5]
                lab_matches = lab_df[lab_df['Sample ID'].str.startswith(prefix, na=False)]
                if not lab_matches.empty:
                    partial_matches.append((clean_spec_id, lab_matches['Sample ID'].iloc[0]))
            
            if partial_matches:
                st.info(f"Found {len(partial_matches)} partial matches")
                for spec_id, lab_id in partial_matches[:5]:
                    st.write(f"- NIR: {spec_id} → Lab: {lab_id}")
        
        # Prepare training data
        dataset, matched_ids, match_type = prepare_training_data(spectra_dict, lab_df)
        if dataset is None:
            st.stop()
        
        # Clean data
        clean_data = clean_and_prepare_data(dataset, TARGET_COLS)
        if clean_data is None:
            st.stop()
        
        # ==============================
        # FEATURE SELECTION SECTION - FIXED
        # ==============================
        st.markdown("### 🎯 Feature Selection Options")
        
        # Get existing targets
        existing_targets = [col for col in TARGET_COLS if col in clean_data.columns]
        
        if not existing_targets:
            st.error("❌ No target columns found in data!")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            feature_selection_method = st.selectbox(
            "Feature Selection Method",
            ["component_based"],  # Only keep component_based
            format_func=lambda x: "🎯 Component-Based"
           )
            # feature_selection_method = st.selectbox(
            #     "Feature Selection Method",
            #     ["component_based", "all_features", "statistical"],
            #     format_func=lambda x: {
            #         "component_based": "🎯 Component-Based",
            #         "all_features": "📊 All Wavelengths",
            #         "statistical": "📈 Statistical Correlation"
            #     }[x]
            # )
        
        with col2:
            if feature_selection_method == "statistical":
                n_features = st.slider("Number of features to select", 10, 500, 100)
            else:
                n_features = None
        
        # Get wavelength columns
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols 
                         if col not in existing_targets + ['Sample ID', 'NIR_Sample_ID', 'Base_Sample_ID']]
        
        # Apply feature selection based on chosen method
        if feature_selection_method == "component_based":
            # Apply COMPONENT-SPECIFIC processing
            st.markdown("### 🎯 Component-Specific Processing")
            
            try:
                # First, let's get the basic data structure
                component_models = {}
                performance_results = []
                
                for target in existing_targets:
                    st.markdown(f"#### 🔧 Processing for **{target}**")
                    
                    # Get preprocessing steps for this component
                    preprocessing_steps = COMPONENT_PREPROCESSING.get(target.lower(), ['snv'])
                    
                    # Select wavelengths for this component
                    selected_wls = []
                    wl_values = [float(w) for w in wavelength_cols]
                    
                    if target.lower() == 'moisture':
                        # Moisture-specific wavelengths
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wl_values, wavelength_cols)
                            if (920 <= wl <= 1000) or (1350 <= wl <= 1550)
                        ]
                        
                    elif target.lower() == 'protein':
                        # Protein-specific wavelengths
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wl_values, wavelength_cols)
                            if (1150 <= wl <= 1300) or (1480 <= wl <= 1650)
                        ]
                        
                    elif target.lower() == 'fat':
                        # FIX: Better fat wavelength selection
                        st.info("🎯 **Fat Analysis**: Using C-H bond absorption regions")
                        
                        # Primary fat absorption regions
                        fat_regions = [
                            (1190, 1210),     # C-H 2nd overtone
                            (1700, 1740),     # C-H 1st overtone (most important)
                            (1350, 1400),     # Associated bands
                            (1650, 1680)      # Adjacent to main C-H region
                        ]
                        
                        selected_wls = []
                        for wl_min, wl_max in fat_regions:
                            for wl, wl_str in zip(wl_values, wavelength_cols):
                                if wl_min <= wl <= wl_max:
                                    selected_wls.append(wl_str)
                        
                        # Remove duplicates
                        selected_wls = list(set(selected_wls))
                        
                        if not selected_wls:
                            st.warning("⚠️ No fat wavelengths found in data range 900-1700nm")
                            # Fallback: use correlation-based selection
                            X_temp = clean_data[wavelength_cols].values.astype(float)
                            y_temp = clean_data[target].values.astype(float)
                            
                            # Calculate correlations
                            correlations = []
                            for i in range(X_temp.shape[1]):
                                corr = np.corrcoef(X_temp[:, i], y_temp)[0, 1]
                                correlations.append(abs(corr))
                            
                            # Select top correlated wavelengths
                            top_n = min(50, len(wavelength_cols))
                            top_idx = np.argsort(correlations)[-top_n:]
                            selected_wls = [wavelength_cols[i] for i in top_idx]
                        
                        # FIX: Use specific preprocessing for fat
                        preprocessing_steps = ['msc', 'savgol', 'snv']
                        
                        st.metric("Fat-specific wavelengths", len(selected_wls))
                        if selected_wls:
                            wl_vals = [float(w) for w in selected_wls]
                            st.caption(f"Range: {min(wl_vals):.0f}-{max(wl_vals):.0f} nm")
                        
                        # Special handling for fat training
                        if selected_wls:
                            X_component = clean_data[selected_wls].values.astype(float)
                            y_component = clean_data[target].values.astype(float)
                            
                            # Check if fat values are reasonable
                            fat_stats = {
                                'mean': y_component.mean(),
                                'std': y_component.std(),
                                'min': y_component.min(),
                                'max': y_component.max()
                            }
                            
                            st.write(f"**Fat Statistics:** Mean={fat_stats['mean']:.2f}, Range={fat_stats['min']:.1f}-{fat_stats['max']:.1f}")
                            
                            # Apply preprocessing
                            X_processed = X_component.copy()
                            for step in preprocessing_steps:
                                X_processed = preprocess_spectra(X_processed, method=step)
                            
                            # Train with more components for fat
                            if len(clean_data) >= 3:
                                test_size = 0.2 if len(clean_data) >= 10 else 0.3
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_processed, y_component, test_size=test_size, random_state=42
                                )
                                
                                # FIX: Use more components for fat (typically needs 5-7)
                                n_components = min(7, X_train.shape[0] - 1, X_train.shape[1])
                                n_components = max(3, n_components)  # At least 3 components
                                
                                st.info(f"Training fat model with {n_components} PLS components")
                                
                                # Train with regularization
                                # Inside the fat training section, after training the model but before storing it:

                                # Train with regularization
                                model = PLSRegression(n_components=n_components)
                                model.fit(X_train, y_train)

                                # EVALUATION AND VALIDATION - ADD THIS HERE:
                                # ===========================================
                                # Validate fat predictions are reasonable
                                y_train_pred = model.predict(X_train)
                                train_mean = y_train_pred.mean()

                                st.write(f"**Training Prediction Stats:** Mean={train_mean:.2f}")

                                # Check if predictions are realistic
                                if train_mean > 30:  # Unrealistically high
                                    st.warning("⚠️ Fat predictions too high - model may need retraining with different wavelengths")
                                    
                                    # Calculate correlation to find better wavelengths
                                    st.info("Finding alternative wavelengths based on correlation...")
                                    
                                    # Calculate correlations for all wavelengths
                                    correlations = []
                                    for i, wl in enumerate(wavelength_cols):
                                        X_temp = clean_data[[wl]].values.astype(float)
                                        corr = np.corrcoef(X_temp.flatten(), y_component)[0, 1]
                                        correlations.append(abs(corr))
                                    
                                    # Select top correlated wavelengths
                                    top_n = min(30, len(wavelength_cols))
                                    top_idx = np.argsort(correlations)[-top_n:]
                                    alternative_wls = [wavelength_cols[i] for i in top_idx]
                                    
                                    if alternative_wls:
                                        st.info(f"Selected {len(alternative_wls)} highly correlated wavelengths")
                                        
                                        # Try training with alternative wavelengths
                                        X_alt = clean_data[alternative_wls].values.astype(float)
                                        
                                        # Apply preprocessing
                                        X_alt_processed = X_alt.copy()
                                        for step in preprocessing_steps:
                                            X_alt_processed = preprocess_spectra(X_alt_processed, method=step)
                                        
                                        # Split data
                                        X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
                                            X_alt_processed, y_component, test_size=test_size, random_state=42
                                        )
                                        
                                        # Train alternative model
                                        model_alt = PLSRegression(n_components=min(5, X_train_alt.shape[1]))
                                        model_alt.fit(X_train_alt, y_train_alt)
                                        
                                        # Check if alternative is better
                                        y_pred_alt = model_alt.predict(X_test_alt)
                                        r2_alt = r2_score(y_test_alt, y_pred_alt)
                                        
                                        y_train_pred_alt = model_alt.predict(X_train_alt)
                                        train_mean_alt = y_train_pred_alt.mean()
                                        
                                        st.write(f"**Alternative model:** R²={r2_alt:.4f}, Mean={train_mean_alt:.2f}")
                                        
                                        # Use the better model
                                        if train_mean_alt < 25 and r2_alt > r2:  # More realistic and better R²
                                            st.success("✅ Alternative model selected (more realistic predictions)")
                                            model = model_alt
                                            selected_wls = alternative_wls
                                        else:
                                            st.info("Keeping original model with bias correction")
                                # ===========================================

                                # Evaluate (continue with existing code)
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                                # Apply bias correction to predictions
                                y_pred_corrected = y_pred * 0.85  # Initial bias correction

                                # Store model with bias correction factor
                                component_models[target] = {
                                    'model': model,
                                    'wavelengths': selected_wls,
                                    'preprocessing': preprocessing_steps,
                                    'n_components': n_components,
                                    'performance': {'R2': r2, 'RMSE': rmse},
                                    'bias_correction': 0.85  # Store correction factor
                                }
                                # model = PLSRegression(n_components=n_components)
                                # model.fit(X_train, y_train)
                                
                                # # Evaluate
                                # y_pred = model.predict(X_test)
                                # r2 = r2_score(y_test, y_pred)
                                # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                
                                # # Apply bias correction to predictions
                                # y_pred_corrected = y_pred * 0.85  # Initial bias correction
                                
                                # # Store model with bias correction factor
                                # component_models[target] = {
                                #     'model': model,
                                #     'wavelengths': selected_wls,
                                #     'preprocessing': preprocessing_steps,
                                #     'n_components': n_components,
                                #     'performance': {'R2': r2, 'RMSE': rmse},
                                #     'bias_correction': 0.85  # Store correction factor
                                # }
                                                
                    elif target.lower() == 'fiber':
                        # Fiber-specific wavelengths
                        selected_wls = [
                            wl_str for wl, wl_str in zip(wl_values, wavelength_cols)
                            if (1050 <= wl <= 1400)
                        ]
                        
                    elif target.lower() == 'ash':
                        # Ash - use all wavelengths
                        selected_wls = wavelength_cols.copy()
                        
                    else:  # wa
                        # For wa, use all wavelengths
                        selected_wls = wavelength_cols.copy()
                    
                    # Show selection info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"Selected Wavelengths", len(selected_wls))
                        if selected_wls:
                            wl_min = min([float(w) for w in selected_wls])
                            wl_max = max([float(w) for w in selected_wls])
                            st.caption(f"Range: {wl_min:.0f}-{wl_max:.0f} nm")
                    with col2:
                        st.metric("Preprocessing", " + ".join(preprocessing_steps))
                    
                    if selected_wls:
                        # Extract X for this component
                        X_component = clean_data[selected_wls].values.astype(float)
                        y_component = clean_data[target].values.astype(float).reshape(-1, 1)
                        
                        # Apply preprocessing steps sequentially
                        X_processed = X_component.copy()
                        for step in preprocessing_steps:
                            X_processed = preprocess_spectra(X_processed, method=step)
                        
                        # Train model for this component
                        if len(clean_data) >= 3:
                            # Split data
                            if len(clean_data) >= 10:
                                test_size = 0.2
                            elif len(clean_data) >= 5:
                                test_size = 0.3
                            else:
                                test_size = 0.4
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_processed, y_component, test_size=test_size, random_state=42
                            )
                            
                            # Determine optimal components
                            n_components = min(10, X_train.shape[0] - 1, X_train.shape[1])
                            n_components = max(1, n_components)
                            
                            # Train PLS model
                            model = PLSRegression(n_components=n_components)
                            model.fit(X_train, y_train)
                            
                            # Evaluate
                            y_pred = model.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            # Store model
                            component_models[target] = {
                                'model': model,
                                'wavelengths': selected_wls,
                                'preprocessing': preprocessing_steps,
                                'performance': {'R2': r2, 'RMSE': rmse, 'MAE': mae}
                            }
                            
                            # Store results
                            performance_results.append({
                                'Component': target,
                                'R²': f"{r2:.4f}",
                                'RMSE': f"{rmse:.4f}",
                                'MAE': f"{mae:.4f}",
                                'Wavelengths': len(selected_wls),
                                'Preprocessing': ' + '.join(preprocessing_steps)
                            })
                            
                            # Show progress
                            status = "✅ Good" if r2 > 0.7 else "⚠️ Needs improvement" if r2 > 0.5 else "❌ Poor"
                            st.info(f"{target}: R²={r2:.4f}, RMSE={rmse:.4f} - {status}")
                        
                # Show performance summary
                if performance_results:
                    st.markdown("### 📊 Model Performance Summary")
                    perf_df = pd.DataFrame(performance_results)
                    st.dataframe(perf_df, use_container_width=True)
                
                # Prepare dataset info for saving
                dataset_info = {
                    'n_samples': len(clean_data),
                    'targets': existing_targets,
                    'created_date': datetime.now().isoformat(),
                    'feature_selection_method': 'component_based'
                }
                
                # Store in session state for prediction
                st.session_state.model_data = {
                    'component_models': component_models,
                    'target_cols': existing_targets,
                    'dataset_info': dataset_info,
                    'preprocessing_info': COMPONENT_PREPROCESSING
                }
                
                # Save model section
                st.markdown("### 💾 Save Component Models")
                if st.button("💾 Save Component Models", type="primary", use_container_width=True):
                    try:
                        model_path = save_component_models(
                            component_models, 
                            model_name, 
                            existing_targets, 
                            dataset_info
                        )
                        
                        st.success(f"✅ Component models saved as `{model_name}`!")
                        st.info(f"📊 Saved {len(component_models)} component-specific models")
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error in component-based processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
        else:  # all_features or statistical method
            # Your original code for non-component-based methods
            st.markdown("### 📊 Using Traditional PLS Model")
            
            # Prepare X and y
            X = clean_data[wavelength_cols].values.astype(float)
            y = clean_data[existing_targets].values.astype(float)
            
            # Apply SNV normalization
            st.markdown("### 🔄 Pre-processing Spectra")
            X_snv = apply_snv(X)
            
            # Check if we have enough data
            if len(clean_data) < 2:
                st.error(f"❌ Not enough samples for training. Only {len(clean_data)} samples available.")
                st.stop()
            
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
            
            # Save model
            st.markdown("### 💾 Save Model")
            if st.button("💾 Save Model", type="primary", use_container_width=True):
                dataset_info = {
                    'n_samples': len(clean_data),
                    'n_wavelengths': len(wavelength_cols),
                    'targets': existing_targets,
                    'feature_selection_method': feature_selection_method,
                    'created_date': datetime.now().isoformat()
                }
                
                try:
                    model_path = save_model(
                        model, 
                        model_name, 
                        wavelength_cols,
                        existing_targets, 
                        dataset_info
                    )
                    
                    st.session_state.trained_model = model
                    st.session_state.model_data = {
                        'model': model,
                        'wavelengths': wavelength_cols,
                        'target_cols': existing_targets,
                        'dataset_info': dataset_info
                    }
                    
                    st.success(f"✅ Model saved as `{model_name}`!")
                    st.info(f"📊 Used {feature_selection_method} with {len(wavelength_cols)} wavelengths")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
    
    else:
        st.info("""
        ## 📚 Training Instructions
        
        **Required Files:**
        1. **IAS 5100 CSV files** - NIR spectral data with Sample ID
        2. **Lab Results file** - CSV or tab-delimited with target values
        
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
                                # FIX: Use model_file instead of selected_model
                                model_path = os.path.join(MODELS_DIR, model_file)
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
                            # FIX: Use model_path (already defined above)
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