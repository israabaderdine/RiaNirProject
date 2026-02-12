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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import io
import pickle
import os
from datetime import datetime
import warnings
import re

# Suppress warnings py -m streamlit run app.py  streamlit run app.py

# Suppress warnings
warnings.filterwarnings('ignore')

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
# TEMPERATURE COMPENSATION MODULE
# ==============================

def detect_temperature_shift(spectrum, reference_water_peak=1940):
    """
    Detect temperature-induced shift in water absorption peak
    
    Parameters:
    - spectrum: pandas Series with wavelength index
    - reference_water_peak: typical water absorption peak (1940 nm for NIR)
    
    Returns:
    - shift_amount: estimated shift in nm
    - temperature_estimate: estimated temperature difference
    """
    # Find local maximum around water peak region
    water_region = spectrum.loc[1900:2000]  # Water peak region
    
    if len(water_region) > 0:
        # Find actual peak
        actual_peak_idx = water_region.idxmax()
        actual_peak = float(actual_peak_idx)
        
        # Calculate shift
        shift_amount = actual_peak - reference_water_peak
        
        # Approximate temperature coefficient
        # For water in NIR, ~0.1-0.2 nm shift per °C
        temp_coefficient = 0.15  # nm/°C
        temp_estimate = shift_amount / temp_coefficient
        
        return shift_amount, temp_estimate
    
    return 0, 0

def temperature_correct_spectrum(spectrum, target_temperature=20, current_temperature=None):
    """
    Apply temperature correction to spectrum
    
    Parameters:
    - spectrum: pandas Series with wavelength index
    - target_temperature: desired temperature for prediction
    - current_temperature: measured temperature (if None, estimate from spectrum)
    
    Returns:
    - corrected_spectrum: temperature-corrected spectrum
    - correction_info: dictionary with correction details
    """
    corrected = spectrum.copy()
    wavelengths = spectrum.index.values
    
    # Convert to numpy array if needed
    if not isinstance(wavelengths, np.ndarray):
        wavelengths = np.array(wavelengths, dtype=float)
    
    # Estimate current temperature if not provided
    if current_temperature is None:
        shift_amount, temp_estimate = detect_temperature_shift(spectrum)
        current_temperature = 20 + temp_estimate  # Assuming reference at 20°C
    else:
        shift_amount = (current_temperature - 20) * 0.15
    
    # Calculate temperature difference
    temp_diff = target_temperature - current_temperature
    temp_shift_correction = temp_diff * 0.15  # nm shift per °C
    
    # Apply wavelength correction
    if abs(temp_shift_correction) > 0.1:
        # Shift wavelengths
        corrected_wavelengths = wavelengths - temp_shift_correction
        
        # Ensure wavelengths are sorted for interpolation
        sorted_indices = np.argsort(corrected_wavelengths)
        corrected_wavelengths_sorted = corrected_wavelengths[sorted_indices]
        spectrum_values_sorted = spectrum.values[sorted_indices]
        
        # Interpolate back to original wavelengths
        corrected_values = np.interp(
            wavelengths, 
            corrected_wavelengths_sorted, 
            spectrum_values_sorted
        )
        corrected = pd.Series(corrected_values, index=wavelengths)
    
    correction_info = {
        'estimated_temperature': current_temperature,
        'target_temperature': target_temperature,
        'wavelength_shift': shift_amount,
        'correction_applied': temp_shift_correction,
        'negative_prediction_risk': abs(temp_diff) > 2  # Risk if >2°C difference
    }
    
    return corrected, correction_info
def predict_with_temperature_compensation(model_data, spectrum, sample_temperature=None):
    """
    Make predictions with automatic temperature compensation
    
    Parameters:
    - model_data: trained model data
    - spectrum: raw spectrum
    - sample_temperature: measured temperature (if available)
    
    Returns:
    - predictions: corrected predictions
    - correction_info: temperature correction details
    - negative_detected: whether negative wa was detected and corrected
    """
    # First, check if water activity might go negative
    _, temp_estimate = detect_temperature_shift(spectrum)
    
    # Apply temperature correction
    corrected_spectrum, correction_info = temperature_correct_spectrum(
        spectrum, 
        target_temperature=20,  # Assume model trained at 20°C
        current_temperature=sample_temperature or (20 + temp_estimate)
    )
    
    # Make prediction with corrected spectrum
    predictions, _ = predict_with_spectrum(model_data, corrected_spectrum)
    
    # Check for negative wa
    negative_detected = False
    if predictions is not None and 'wa' in predictions.columns:
        wa_value = predictions['wa'].iloc[0]
        if wa_value < 0:
            negative_detected = True
            # Apply additional constraint for water activity (must be 0-1)
            predictions['wa'] = predictions['wa'].clip(lower=0, upper=1)
            correction_info['negative_corrected'] = True
    
    return predictions, correction_info, negative_detected

def predict_with_robustness_check(model_data, spectrum, sample_temperature=None):
    """Enhanced prediction with negative value detection and correction"""
    
    # Use temperature-compensated prediction
    predictions, correction_info, negative_detected = predict_with_temperature_compensation(
        model_data, spectrum, sample_temperature
    )
    
    if negative_detected:
        st.warning(f"""
        ⚠️ **Temperature-Induced Negative Water Activity Detected**
        
        **Cause:** Sample temperature differs from model training temperature
        - Estimated sample temperature: {correction_info['estimated_temperature']:.1f}°C
        - Model training temperature: {correction_info['target_temperature']:.1f}°C
        - Wavelength shift detected: {correction_info['wavelength_shift']:.2f} nm
        
        **Correction Applied:**
        - Spectrum shifted by {correction_info['correction_applied']:.2f} nm
        - Water activity constrained to physical range (0-1)
        """)
        
        # Show the shift visually
        fig = visualize_temperature_shift(spectrum, correction_info)
        st.plotly_chart(fig, use_container_width=True)
    
    return predictions, correction_info
def augment_with_temperature_variations(X, y, wavelengths, temperatures=None):
    """
    Augment training data with simulated temperature variations
    
    Parameters:
    - X: original spectra
    - y: target values
    - wavelengths: wavelength points (list or array)
    - temperatures: measured temperatures (or None to simulate)
    
    Returns:
    - X_aug, y_aug: augmented dataset with temperature variations
    """
    augmented_spectra = []
    augmented_targets = []
    
    # Temperature variations to simulate (±0°C to ±5°C)
    temp_variations = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
    
    # Convert wavelengths to numpy array for mathematical operations
    wavelengths_array = np.array(wavelengths, dtype=float)
    
    for i in range(X.shape[0]):
        spectrum = pd.Series(X[i], index=wavelengths_array)
        
        for temp_shift in temp_variations:
            # Simulate temperature-induced shift
            shifted_wavelengths = wavelengths_array - (temp_shift * 0.15)
            
            # Interpolate - ensure shifted_wavelengths is sorted
            # np.interp requires x-coordinates to be increasing
            sorted_indices = np.argsort(shifted_wavelengths)
            shifted_wavelengths_sorted = shifted_wavelengths[sorted_indices]
            spectrum_values_sorted = spectrum.values[sorted_indices]
            
            # Interpolate back to original wavelengths
            shifted_values = np.interp(
                wavelengths_array, 
                shifted_wavelengths_sorted, 
                spectrum_values_sorted
            )
            
            augmented_spectrum = pd.Series(shifted_values, index=wavelengths_array)
            
            # Add small noise
            noise = np.random.normal(0, 0.0005, len(wavelengths_array))
            augmented_spectrum = augmented_spectrum + noise
            
            augmented_spectra.append(augmented_spectrum.values)
            augmented_targets.append(y[i])
    
    # Stack arrays
    if augmented_spectra:
        X_aug = np.vstack([X] + augmented_spectra)
        y_aug = np.vstack([y] + augmented_targets)
    else:
        X_aug = X
        y_aug = y
    
    return X_aug, y_aug
def visualize_temperature_shift(spectrum, correction_info):
    """Visualize the temperature-induced spectral shift"""
    fig = go.Figure()
    
    # Original spectrum
    fig.add_trace(go.Scatter(
        x=spectrum.index,
        y=spectrum.values,
        mode='lines',
        name=f'Original Spectrum ({correction_info["estimated_temperature"]:.1f}°C)',
        line=dict(color='#EF4444', width=2)
    ))
    
    # Corrected spectrum
    corrected, _ = temperature_correct_spectrum(spectrum)
    fig.add_trace(go.Scatter(
        x=corrected.index,
        y=corrected.values,
        mode='lines',
        name=f'Corrected Spectrum ({correction_info["target_temperature"]:.1f}°C)',
        line=dict(color='#10B981', width=2, dash='dot')
    ))
    
    # Highlight water peak region
    fig.add_vrect(
        x0=1930, x1=1950,
        fillcolor="#764ba2", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Water Peak Region",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f"Temperature Compensation - Shift: {correction_info['correction_applied']:.2f} nm",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        template="plotly_white",
        height=400
    )
    
    return fig
def augment_spectrum(spectrum, noise_level=0.001, shift_max=1, scaling_range=(0.98, 1.02)):
    """
    Apply multiple augmentation techniques to a single spectrum
    
    Parameters:
    - spectrum: pandas Series with wavelength index
    - noise_level: standard deviation of Gaussian noise
    - shift_max: maximum wavelength shift in nm
    - scaling_range: min/max for multiplicative scaling
    """
    augmented = spectrum.copy()
    
    # 1. Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(spectrum))
        augmented = augmented + noise
    
    # 2. Multiplicative scaling
    scale_factor = np.random.uniform(scaling_range[0], scaling_range[1])
    augmented = augmented * scale_factor
    
    # 3. Baseline shift
    baseline_shift = np.random.normal(0, noise_level * 10)
    augmented = augmented + baseline_shift
    
    return augmented

def augment_with_warping(spectrum, warp_factor=0.01):
    """
    Apply wavelength warping (small shifts in wavelength domain)
    """
    wavelengths = spectrum.index.values
    new_wavelengths = wavelengths * (1 + np.random.uniform(-warp_factor, warp_factor))
    
    # Interpolate back to original wavelengths
    warped_spectrum = pd.Series(
        np.interp(wavelengths, new_wavelengths, spectrum.values),
        index=wavelengths
    )
    
    return warped_spectrum
def augment_with_mixup(spectrum1, spectrum2, y1, y2, alpha=0.2):
    """
    MixUp augmentation: create weighted combination of two spectra and their labels
    
    Parameters:
    - spectrum1, spectrum2: pandas Series with wavelength index
    - y1, y2: target values for each spectrum
    - alpha: Beta distribution parameter
    """
    lambda_val = np.random.beta(alpha, alpha)
    
    # Mix spectra
    mixed_spectrum = lambda_val * spectrum1 + (1 - lambda_val) * spectrum2
    
    # Mix labels
    mixed_y = lambda_val * y1 + (1 - lambda_val) * y2
    
    return mixed_spectrum, mixed_y

def augment_with_warping(spectrum, warp_factor=0.01):
    """
    Apply wavelength warping (small shifts in wavelength domain)
    """
    wavelengths = spectrum.index.values
    new_wavelengths = wavelengths * (1 + np.random.uniform(-warp_factor, warp_factor))
    
    # Interpolate back to original wavelengths
    warped_spectrum = pd.Series(
        np.interp(wavelengths, new_wavelengths, spectrum.values),
        index=wavelengths
    )
    
    return warped_spectrum
def create_augmented_dataset(X, y, wavelengths, sample_ids, augmentation_factor=3, 
                            use_noise=True, use_warp=True, use_mixup=True, use_scale=True,
                            noise_level=0.001):
    """
    Create augmented dataset with multiple techniques
    
    Parameters:
    - X: original spectra matrix (n_samples, n_features)
    - y: target values matrix (n_samples, n_targets)
    - wavelengths: list of wavelength values
    - sample_ids: list of sample IDs
    - augmentation_factor: multiplier for dataset size
    - use_noise: whether to use noise augmentation
    - use_warp: whether to use wavelength warping
    - use_mixup: whether to use MixUp augmentation
    - use_scale: whether to use scaling augmentation
    - noise_level: standard deviation for Gaussian noise
    
    Returns:
    - X_aug, y_aug, augmented_ids
    """
    augmented_spectra = []
    augmented_targets = []
    augmented_ids = []
    
    n_samples = X.shape[0]
    n_targets = y.shape[1] if len(y.shape) > 1 else 1
    
    # Collect available methods
    available_methods = []
    if use_noise:
        available_methods.append('noise')
    if use_warp:
        available_methods.append('warp')
    if use_mixup:
        available_methods.append('mixup')
    if use_scale:
        available_methods.append('scale')
    
    # Default to noise if no methods selected
    if not available_methods:
        available_methods = ['noise']
    
    # Create augmented samples
    for i in range(n_samples):
        # Original spectrum as Series
        spectrum = pd.Series(X[i], index=wavelengths)
        
        # Create multiple augmented versions
        for aug_idx in range(augmentation_factor):
            # Randomly choose augmentation method
            method = np.random.choice(available_methods)
            
            if method == 'noise':
                aug_spectrum = augment_spectrum(
                    spectrum, 
                    noise_level=noise_level
                )
                aug_y = y[i].copy()
                
            elif method == 'warp':
                aug_spectrum = augment_with_warping(
                    spectrum,
                    warp_factor=np.random.uniform(0.005, 0.02)
                )
                aug_y = y[i].copy()
                
            elif method == 'mixup':
                # Mix with another random sample
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                
                spectrum2 = pd.Series(X[j], index=wavelengths)
                aug_spectrum, aug_y = augment_with_mixup(
                    spectrum, spectrum2,
                    y[i], y[j],
                    alpha=0.2
                )
                
            elif method == 'scale':
                # Scaling augmentation
                aug_spectrum = augment_spectrum(
                    spectrum,
                    noise_level=0,  # No noise
                    scaling_range=(0.95, 1.05)
                )
                aug_y = y[i].copy()
            
            # Ensure no NaN values
            if aug_spectrum.isna().any():
                aug_spectrum = aug_spectrum.interpolate(method='linear')
                aug_spectrum = aug_spectrum.bfill().ffill()
            
            augmented_spectra.append(aug_spectrum.values)
            augmented_targets.append(aug_y)
            augmented_ids.append(f"{sample_ids[i]}_aug{aug_idx+1}")
    
    # Combine original and augmented data
    X_aug = np.vstack([X, np.array(augmented_spectra)])
    y_aug = np.vstack([y, np.array(augmented_targets)])
    all_ids = list(sample_ids) + augmented_ids
    
    return X_aug, y_aug, all_ids

def plot_augmentation_example(original_spectrum, augmented_spectra, sample_id):
    """Plot original vs augmented spectra for visualization"""
    fig = go.Figure()
    
    # Original spectrum
    fig.add_trace(go.Scatter(
        x=original_spectrum.index,
        y=original_spectrum.values,
        mode='lines',
        name=f'Original {sample_id}',
        line=dict(color='#764ba2', width=3)
    ))
    
    # Augmented spectra
    colors = ['#10B981', '#F59E0B', '#EF4444', '#3B82F6']
    for idx, (name, aug_spec) in enumerate(augmented_spectra.items()):
        fig.add_trace(go.Scatter(
            x=aug_spec.index,
            y=aug_spec.values,
            mode='lines',
            name=name,
            line=dict(color=colors[idx % len(colors)], width=1.5, dash='dot'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=f"Data Augmentation Example - {sample_id}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        template="plotly_white",
        hovermode='x unified',
        height=500
    )
    
    return fig








def parse_ias_5100(file, duplicate_counter=None):
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
            
            # Apply duplicate suffix if provided
            display_id = sample_id
            if duplicate_counter and duplicate_counter > 1:
                display_id = f"{sample_id}_dup{duplicate_counter}"
                st.info(f"📝 Duplicate sample ID '{sample_id}' found. Using '{display_id}' for this spectrum.")
            
            st.success(f"✅ {file_name}: ID={display_id}, Points={len(spectrum)}, Range={spectrum.index.min():.0f}-{spectrum.index.max():.0f}nm")
            return spectrum, display_id
        
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
    """Apply different spectral pre-processing methods"""
    if method == 'snv':
        return apply_snv(X)
    elif method == 'msc':
        # Multiplicative Scatter Correction
        mean_spectrum = np.mean(X, axis=0)
        for i in range(X.shape[0]):
            # Fit each spectrum to mean spectrum
            coeffs = np.polyfit(mean_spectrum, X[i], 1)
            X[i] = (X[i] - coeffs[1]) / coeffs[0]
        return X
    elif method == 'derivative':
        # First derivative
        return np.gradient(X, axis=1)
    elif method == 'savgol':
        # Savitzky-Golay smoothing + derivative
        from scipy.signal import savgol_filter
        X_smooth = savgol_filter(X, window_length=11, polyorder=2, axis=1)
        return np.gradient(X_smooth, axis=1)
    return X
def get_model_selector():
    """
    إنشاء واجهة اختيار النموذج للمستخدم
    """
    st.markdown("### 🤖 Model Selection")
    st.markdown("اختر نوع النموذج الذي تريد تدريبه:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "**نوع النموذج:**",
            [
                "📊 PLS Regression (سريع - خطي)",
                "🌲 Random Forest (دقيق - غير خطي)", 
                "⚡ XGBoost (الأفضل - دقة عالية)"
            ],
            index=0  # PLS هو الاختيار الافتراضي
        )
    
    with col2:
        st.markdown("**مقارنة سريعة:**")
        if "PLS" in model_type:
            st.info("✅ مناسب للعينات القليلة\n✅ سريع جداً\n⚠️ دقة متوسطة\n❌ يحتاج تطبيع البيانات")
        elif "Random Forest" in model_type:
            st.info("✅ دقة عالية\n✅ لا يحتاج تطبيع\n✅ يعطيك أهمية المتغيرات\n⚠️ بطيء قليلاً")
        elif "XGBoost" in model_type:
            st.info("✅ أعلى دقة\n✅ سريع جداً\n✅ يتعامل مع القيم المفقودة\n⚠️ يحتاج تعديل بارامترات")
    
    # بارامترات إضافية حسب النموذج
    params = {}
    
    if "Random Forest" in model_type:
        st.markdown("---")
        st.markdown("**🔧 إعدادات Random Forest:**")
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider(
                "عدد الأشجار",
                min_value=50, max_value=500, value=200, step=50,
                help="زيادة العدد = دقة أعلى لكن أبطأ"
            )
            params['max_depth'] = st.slider(
                "عمق الشجرة",
                min_value=5, max_value=30, value=15, step=5,
                help="عمق أكبر = نموذج أكثر تعقيداً"
            )
        with col2:
            params['min_samples_split'] = st.slider(
                "أقل عدد للتقسيم",
                min_value=2, max_value=10, value=5, step=1,
                help="يمنع overfitting"
            )
            params['min_samples_leaf'] = st.slider(
                "أقل عدد في الورقة",
                min_value=1, max_value=5, value=2, step=1,
                help="يمنع overfitting"
            )
            
    elif "XGBoost" in model_type:
        st.markdown("---")
        st.markdown("**🔧 إعدادات XGBoost:**")
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider(
                "عدد التكرارات",
                min_value=50, max_value=500, value=200, step=50
            )
            params['learning_rate'] = st.select_slider(
                "معدل التعلم",
                options=[0.01, 0.05, 0.1, 0.2, 0.3],
                value=0.1
            )
        with col2:
            params['max_depth'] = st.slider(
                "العمق الأقصى",
                min_value=3, max_value=15, value=6, step=1
            )
            params['subsample'] = st.select_slider(
                "نسبة العينات",
                options=[0.6, 0.7, 0.8, 0.9, 1.0],
                value=0.8
            )
    
    return model_type, params
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

def save_model(model, model_name, wavelengths, target_cols, dataset_info, model_type):
    """Save trained model to file"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # تحديد نوع النموذج
    if 'PLSRegression' in str(type(model)):
        model_type_name = 'PLS'
    elif 'RandomForest' in str(type(model)):
        model_type_name = 'Random Forest'
    elif 'XGBoost' in str(type(model)):
        model_type_name = 'XGBoost'
    else:
        model_type_name = 'Unknown'
    
    model_data = {
        'model': model,
        'wavelengths': wavelengths,
        'target_cols': target_cols,
        'model_type': model_type_name,
        'created_date': datetime.now().isoformat(),
        'dataset_info': dataset_info
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path
def predict_with_spectrum(model_data, spectrum):
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
            spectrum_aligned = spectrum_aligned.bfill().ffill()
        
        # Prepare for prediction
        X_new = spectrum_aligned.values.reshape(1, -1)
        
        # Check if model is PLS (needs SNV) or Tree-based (doesn't need)
        model_type = type(model).__name__
        
        if 'PLSRegression' in str(model_type):
            X_new_processed = apply_snv(X_new)
        else:
            # Random Forest and XGBoost don't need normalization
            X_new_processed = X_new
        
        # Make prediction
        prediction = model.predict(X_new_processed)
        
        # Clip negative values to 0
        prediction = np.maximum(prediction, 0)
        
        # Create result DataFrame
        result_df = pd.DataFrame(prediction, columns=target_cols)
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
        duplicate_counter = {}  # Track duplicates per sample ID

        for idx, file in enumerate(nir_files):
            with st.spinner(f"Parsing {file.name}..."):
                # Check if this sample ID already exists
                # First parse without storing to get the base ID
                temp_spectrum, temp_id = parse_ias_5100(file)
                
                if not temp_spectrum.empty:
                    base_id = re.sub(r'_dup\d+$', '', temp_id)  # Remove any existing _dup suffix
                    
                    # Update counter for this base ID
                    if base_id in duplicate_counter:
                        duplicate_counter[base_id] += 1
                    else:
                        duplicate_counter[base_id] = 1
                    
                    # Now parse with the correct duplicate counter
                    spectrum, unique_id = parse_ias_5100(file, duplicate_counter[base_id])
                    spectra_dict[unique_id] = spectrum
                
            progress_bar.progress((idx + 1) / len(nir_files))

        # Count total duplicates
        total_duplicates = sum(count - 1 for count in duplicate_counter.values() if count > 1)
        if total_duplicates > 0:
            st.warning(f"⚠️ Found {total_duplicates} duplicate sample IDs. Each spectrum will be used separately for training with _dup suffix.")
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
        
        # Extract training data
        existing_targets = [col for col in TARGET_COLS if col in clean_data.columns]
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
        wavelength_cols = [col for col in numeric_cols if col not in existing_targets + ['Sample ID']]

        # Check if we have enough data
        if len(clean_data) < 2:
            st.error(f"❌ Not enough samples for training. Only {len(clean_data)} samples available.")
            st.stop()

        X = clean_data[wavelength_cols].values.astype(float)
        y = clean_data[existing_targets].values.astype(float)
        sample_ids = clean_data['Sample ID'].tolist() if 'Sample ID' in clean_data.columns else [f"sample_{i}" for i in range(len(clean_data))]

        # Initialize training data arrays
        X_train_full = X
        y_train_full = y

        # ============================================
        # DATA AUGMENTATION SECTION
        # ============================================
        st.markdown("### 🔬 Data Augmentation")
        st.markdown("Enhance your dataset with synthetic spectra to improve model robustness")

        col1, col2, col3 = st.columns(3)

        with col1:
            use_augmentation = st.checkbox("✅ Enable Data Augmentation", value=False, key="use_aug")
            
        with col2:
            if use_augmentation:
                augmentation_factor = st.slider(
                    "Augmentation Multiplier",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Number of synthetic spectra to generate per original sample"
                )
            else:
                augmentation_factor = 0
                
        with col3:
            if use_augmentation:
                noise_level = st.slider(
                    "Noise Level",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    format="%.4f",
                    help="Standard deviation of Gaussian noise"
                )
            else:
                noise_level = 0.001

        # Temperature Variation Augmentation (ADD THIS NEW SECTION)
        if use_augmentation:
            with st.expander("🌡️ Temperature Compensation Training", expanded=True):
                st.markdown("""
                **Why this matters:** Your data shows that a 2°C temperature difference 
                shifts the water peak and causes negative predictions.
                
                Enable temperature-aware training to make your model robust to 
                temperature variations.
                """)
                
                use_temperature_aug = st.checkbox(
                    "✅ Enable Temperature Variation Training",
                    value=True,
                    help="Train model with simulated temperature variations (±5°C)"
                )
                
                if use_temperature_aug:
                    wavelengths_float = [float(w) for w in wavelength_cols]
                    
                    # Create temperature-augmented dataset
                    X_temp_aug, y_temp_aug = augment_with_temperature_variations(
                        X, y, wavelengths_float
                    )
                    
                    st.success(f"✅ Added {len(X_temp_aug) - len(X)} temperature-varied spectra")
                    
                    # Combine with existing data
                    X_train_full = np.vstack([X_train_full, X_temp_aug])
                    y_train_full = np.vstack([y_train_full, y_temp_aug])
                    
                    st.metric("📊 Total Training Samples", len(X_train_full))

        # Show augmentation example (EXISTING CODE - KEEP THIS)
        if use_augmentation and len(clean_data) > 0:
            with st.expander("📊 Augmentation Preview", expanded=True):
                # Select a random sample for demonstration
                demo_idx = np.random.randint(0, len(clean_data))
                demo_spectrum = pd.Series(X[demo_idx], index=[float(w) for w in wavelength_cols])
                demo_id = sample_ids[demo_idx]
                
                # Generate examples
                demo_augmented = {}
                
                # Noise augmentation
                aug_spec_noise = augment_spectrum(demo_spectrum, noise_level=noise_level)
                demo_augmented[f'Noise (σ={noise_level})'] = aug_spec_noise
                
                # Warping augmentation
                aug_spec_warp = augment_with_warping(demo_spectrum, warp_factor=0.01)
                demo_augmented['Warping'] = aug_spec_warp
                
                # MixUp augmentation
                j = (demo_idx + 1) % len(clean_data)
                spec2 = pd.Series(X[j], index=[float(w) for w in wavelength_cols])
                aug_spec_mix, _ = augment_with_mixup(demo_spectrum, spec2, y[demo_idx], y[j])
                demo_augmented['MixUp'] = aug_spec_mix
                
                # Scaling augmentation
                aug_spec_scale = augment_spectrum(demo_spectrum, scaling_range=(0.95, 1.05))
                demo_augmented['Scaling'] = aug_spec_scale
                
                # Plot example
                fig = plot_augmentation_example(demo_spectrum, demo_augmented, demo_id)
                st.plotly_chart(fig, use_container_width=True)
                
                # Method selection
                st.markdown("**Select Augmentation Methods:**")
                aug_methods_col1, aug_methods_col2, aug_methods_col3, aug_methods_col4 = st.columns(4)
                
                with aug_methods_col1:
                    use_noise = st.checkbox("Noise", value=False)
                with aug_methods_col2:
                    use_warp = st.checkbox("Warping", value=False)
                with aug_methods_col3:
                    use_mixup = st.checkbox("MixUp", value=False)
                with aug_methods_col4:
                    use_scale = st.checkbox("Scaling", value=True)
                
                st.info("""
                **Augmentation Methods:**
                - **Noise**: Adds Gaussian noise and baseline variations
                - **Warping**: Small shifts in wavelength domain  
                - **MixUp**: Creates weighted combinations of two spectra
                - **Scaling**: Multiplicative scaling of absorbance values
                """)
        else:
            use_noise = True
            use_warp = True
            use_mixup = True
            use_scale = True



        # Apply augmentation if enabled
        if use_augmentation and augmentation_factor > 0:
            with st.spinner("🔄 Creating augmented dataset..."):
                wavelengths_float = [float(w) for w in wavelength_cols]
                
                # Create augmented dataset with selected methods
                X_aug, y_aug, augmented_ids = create_augmented_dataset(
                    X, y, 
                    wavelengths_float, 
                    sample_ids,
                    augmentation_factor=augmentation_factor,
                    use_noise=use_noise,
                    use_warp=use_warp,
                    use_mixup=use_mixup,
                    use_scale=use_scale,
                    noise_level=noise_level
                )
                
                # Display augmentation stats
                aug_col1, aug_col2, aug_col3, aug_col4 = st.columns(4)
                with aug_col1:
                    st.metric("📊 Original Samples", len(X))
                with aug_col2:
                    st.metric("🔄 Augmented Samples", len(X_aug) - len(X))
                with aug_col3:
                    st.metric("🎯 Total Samples", len(X_aug))
                with aug_col4:
                    augmentation_ratio = (len(X_aug) - len(X)) / len(X) * 100
                    st.metric("📈 Augmentation Ratio", f"{augmentation_ratio:.0f}%")
                
                # Use augmented data for training
                X_train_full = X_aug
                y_train_full = y_aug
                st.success(f"✅ Dataset augmented from {len(X)} to {len(X_aug)} samples!")
                
                # Show sample of augmented IDs
                with st.expander("🔍 View Augmented Sample IDs"):
                    st.write("**First 20 augmented samples:**")
                    st.write(augmented_ids[:20])
        else:
            X_train_full = X
            y_train_full = y
            st.info("ℹ️ Using original dataset without augmentation")

        # Apply SNV normalization
        X_snv = apply_snv(X_train_full)

        # Train model
        st.markdown("### 🤖 Training Model")

        # Adjust PLS components based on augmented data
        n_components = min(15, X_snv.shape[0] - 1, X_snv.shape[1])  # Increased max components
        n_components = max(1, n_components)

        # Split data - use original indices for test set if augmented
        if use_augmentation:
            # Use only original samples for test set to avoid data leakage
            test_size = min(0.2, max(0.05, 1/len(X)))
            
            # Ensure test_size is valid
            if test_size >= 1.0:
                test_size = 0.2
            
            X_train, X_test, y_train, y_test = train_test_split(
                X[:len(X)], y[:len(X)], test_size=test_size, random_state=42
            )
            
            # Combine original training data with augmented data for training
            X_train_final = np.vstack([X_train, X_aug[len(X):]])
            y_train_final = np.vstack([y_train, y_aug[len(X):]])
            
        else:
            test_size = min(0.2, max(0.05, 1/len(clean_data)))
            if test_size >= 1.0:
                test_size = 0.2
                
            X_train, X_test, y_train, y_test = train_test_split(
                X_snv, y, test_size=test_size, random_state=42
            )
            X_train_final = X_train
            y_train_final = y_train

        # Apply SNV to training data
        if use_augmentation:
            X_train_snv = apply_snv(X_train_final)
            X_test_snv = apply_snv(X_test)
        else:
            X_train_snv = X_train
            X_test_snv = X_test

        # ============================================
        # MODEL SELECTION SECTION
        # ============================================
        model_type, model_params = get_model_selector()
        
        # Train model based on selection
        st.markdown("### 🤖 Training Model")
        
        with st.spinner(f"Training {model_type.split('(')[0].strip()} model..."):
            
            # اختيار النموذج المناسب
            if "Random Forest" in model_type:
                # Random Forest لا يحتاج تطبيع
                if use_augmentation:
                    X_train_final_rf = X_train_final
                    X_test_rf = X_test
                else:
                    X_train_final_rf = X_train
                    X_test_rf = X_test
                
                # تدريب Random Forest مع MultiOutput
                base_model = RandomForestRegressor(
                    n_estimators=model_params.get('n_estimators', 200),
                    max_depth=model_params.get('max_depth', 15),
                    min_samples_split=model_params.get('min_samples_split', 5),
                    min_samples_leaf=model_params.get('min_samples_leaf', 2),
                    random_state=42,
                    n_jobs=-1
                )
                model = MultiOutputRegressor(base_model)
                model.fit(X_train_final_rf, y_train_final)
                
                # للتقييم
                X_test_eval = X_test_rf
                
            elif "XGBoost" in model_type:
                # XGBoost لا يحتاج تطبيع
                if use_augmentation:
                    X_train_final_xgb = X_train_final
                    X_test_xgb = X_test
                else:
                    X_train_final_xgb = X_train
                    X_test_xgb = X_test
                
                # تدريب XGBoost مع MultiOutput
                base_model = XGBRegressor(
                n_estimators=model_params.get('n_estimators', 200),
                learning_rate=model_params.get('learning_rate', 0.1),
                max_depth=model_params.get('max_depth', 6),
                subsample=model_params.get('subsample', 0.8),
                random_state=42,
                n_jobs=-1,
                tree_method="hist"   # CPU safe
                )

                # base_model = XGBRegressor(
                #     n_estimators=model_params.get('n_estimators', 200),
                #     learning_rate=model_params.get('learning_rate', 0.1),
                #     max_depth=model_params.get('max_depth', 6),
                #     subsample=model_params.get('subsample', 0.8),
                #     random_state=42,
                #     n_jobs=-1
                # )
                model = MultiOutputRegressor(base_model)
                model.fit(X_train_final_xgb, y_train_final)
                
                # للتقييم
                X_test_eval = X_test_xgb
                
            else:  # PLS
                # PLS يحتاج تطبيع
                X_train_snv = apply_snv(X_train_final)
                X_test_snv = apply_snv(X_test)
                
                model = PLSRegression(n_components=n_components)
                model.fit(X_train_snv, y_train_final)
                
                # للتقييم
                X_test_eval = X_test_snv
        
        # اسم النموذج للعرض
        model_display_name = model_type.split('(')[0].strip()
        st.success(f"✅ {model_display_name} model trained successfully!")
        
        # ============================================
        # MODEL EVALUATION SECTION
        # ============================================
        st.markdown("### 📈 Model Performance Evaluation")
        
        # Make predictions on test set
        y_pred = model.predict(X_test_eval)
        y_pred = np.maximum(y_pred, 0)  # Clip negative values
        
        # Calculate metrics for each target
        eval_cols = st.columns(len(existing_targets))
        metrics_data = []
        
        for i, target in enumerate(existing_targets):
            with eval_cols[i]:
                if len(np.unique(y_test[:, i])) > 1:
                    # Calculate metrics
                    test_r2 = r2_score(y_test[:, i], y_pred[:, i])
                    test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                    test_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                    test_mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-10))) * 100
                    
                  # Status emoji
                    if test_r2 > 0.8:
                        status = "🏆 Excellent"
                    elif test_r2 > 0.7:
                        status = "✅ Good"  
                    elif test_r2 > 0.5:
                        status = "⚠️ Fair"
                    else:
                        status = "❌ Poor"
                    
                    metrics_data.append({
                        'Target': target,
                        'Test R²': f"{test_r2:.4f}",
                        'Test RMSE': f"{test_rmse:.4f}",
                        'Test MAE': f"{test_mae:.4f}",
                        'Test MAPE': f"{test_mape:.1f}%",
                        'Status': status
                    })
                    
                    # Create gauge for R²
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=test_r2,
                        title={'text': f"{target} R²"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "#764ba2"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "#EF4444"},
                                {'range': [0.5, 0.7], 'color': "#F59E0B"},
                                {'range': [0.7, 0.8], 'color': "#10B981"},
                                {'range': [0.8, 1], 'color': "#059669"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
        
        # عرض المقاييس في جدول
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # ============================================
            # FEATURE IMPORTANCE (لـ Random Forest فقط)
            # ============================================
            if "Random Forest" in model_type and hasattr(model, 'estimators_'):
                with st.expander("🌲 أهمية الأطوال الموجية (Feature Importance)", expanded=True):
                    st.markdown("""
                    **هذه أهم الأطوال الموجية في توقع المكونات:**
                    كلما زاد الطول، كلما كان أكثر أهمية للنموذج.
                    """)
                    
                    # حساب متوسط الأهمية لكل الأهداف
                    all_importances = []
                    for estimator in model.estimators_:
                        all_importances.append(estimator.feature_importances_)
                    
                    avg_importance = np.mean(all_importances, axis=0)
                    
                    # ترتيب الأطوال الموجية حسب الأهمية
                    wavelengths_float = [float(w) for w in wavelength_cols]
                    importance_df = pd.DataFrame({
                        'Wavelength (nm)': wavelengths_float,
                        'Importance': avg_importance
                    }).sort_values('Importance', ascending=False)
                    
                    # عرض أهم 20 طول موجي
                    top_n = min(20, len(importance_df))
                    st.markdown(f"**🔝 أهم {top_n} طول موجي:**")
                    
                    fig_importance = px.bar(
                        importance_df.head(top_n),
                        x='Wavelength (nm)',
                        y='Importance',
                        title='Feature Importance - Top Wavelengths',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # عرض في جدول
                    st.dataframe(importance_df.head(10), use_container_width=True)
        # Save model
        st.markdown("### 💾 Save Model")
        
        model_name_input = st.text_input(
            "Model Name",
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Give your model a descriptive name"
        )
        
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            if st.button("💾 Save Model", type="primary", use_container_width=True):
                dataset_info = {
                    'n_samples': len(clean_data),
                    'n_augmented_samples': len(X_train_full) - len(clean_data) if use_augmentation else 0,
                    'n_wavelengths': len(wavelength_cols),
                    'targets': existing_targets,
                    'sample_ids': clean_data['Sample ID'].tolist() if 'Sample ID' in clean_data.columns else [],
                    'created_date': datetime.now().isoformat(),
                    'augmentation_used': use_augmentation,
                    'augmentation_factor': augmentation_factor if use_augmentation else 0,
                    'model_metrics': metrics_data if metrics_data else []
                }
                
                model_path = save_model(model, model_name_input, wavelength_cols, existing_targets, dataset_info, model_type)
                st.session_state.trained_model = model
                st.session_state.model_data = {
                    'model': model,
                    'wavelengths': wavelength_cols,
                    'target_cols': existing_targets,
                    'dataset_info': dataset_info
                }
                
                st.success(f"✅ Model saved as `{model_name_input}`!")
                
        with col_save2:
            if st.button("🔄 Train Another Model", use_container_width=True):
                st.rerun()
    
    else:
        # Training instructions when no files uploaded
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 2rem; border-radius: 12px;'>
            <h2 style='color: #764ba2;'>📚 Training Instructions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        inst_col1, inst_col2 = st.columns(2)
        
        with inst_col1:
            st.markdown("""
            ### 📁 Required Files
            
            **1. IAS 5100 CSV files**
            - NIR spectral data with Sample ID
            - Format similar to IAS_Spectrum.csv
            - Supports multiple files
            
            **2. Lab Results file**
            - CSV or tab-delimited format
            - Must contain Sample ID column
            - Target columns: Protein, fat, ash, moisture, Fiber, wa
            """)
            
        with inst_col2:
            st.markdown("""
            ### 🎯 Features
            
            **Data Processing:**
            - Automatic Sample ID matching
            - Duplicate handling with _dup suffix
            - Missing value imputation
            - SNV normalization
            
            **Data Augmentation:**
            - Gaussian noise injection
            - Wavelength warping
            - MixUp combinations
            - Scaling augmentation
            
            **Model Training:**
            - PLS regression
            - Automatic component selection
            - Train/test split
            - Performance metrics
            """)
        
        st.markdown("### 📋 Lab File Format Example")
        
        example_data = {
            'Sample ID': ['B2600041_01', 'B2600041_02', 'B2500012'],
            'Protein': [23, 23, 30],
            'fat': [21, 21, 13],
            'ash': [4.6, 4.6, 5.6],
            'moisture': [5, 5, 6.3],
            'Fiber': [3.49, 3.49, 3.35],
            'wa': [0.4, 0.4, 0.47]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.info("""
        👆 **Ready to start?** Upload your NIR files and lab results file using the sidebar to begin training.
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
                    if st.session_state.model_data:
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
                                        # FIX: Add unique key using file name and column name
                                        st.plotly_chart(
                                            fig, 
                                            use_container_width=True,
                                            key=f"gauge_{file.name}_{col_name}_{idx}"
                                        )
                                    
                            # Add to summary
                            prediction_row = result_df.copy()
                            prediction_row['File'] = file.name
                            prediction_row['Sample ID'] = sn
                            all_predictions.append(prediction_row)
                            
                            # Plot spectrum
                            st.markdown("### 📈 Spectrum")
                            fig = go.Figure()
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
                            st.session_state.model_data = model_data
                            st.session_state.trained_model = model_data['model']
                            st.success("✅ Model loaded!")
                            st.rerun()
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
    
    if st.session_state.model_data:
        st.markdown("---")
        st.markdown("### ✅ Currently Loaded Model")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Target Parameters", len(st.session_state.model_data['target_cols']))
            st.write("**Parameters:**")
            for target in st.session_state.model_data['target_cols']:
                st.markdown(f"• {target}")
        with col2:
            st.metric("Wavelength Points", len(st.session_state.model_data['wavelengths']))
            st.metric("Status", "Ready for predictions ✅")

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