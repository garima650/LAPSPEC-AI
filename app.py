import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================
# 1. CORE VISUAL STYLING ENGINE (NEON AMBER)
# ==========================================
st.set_page_config(
    page_title="Valuation Core Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium matte amber & gunmetal steel styling overrides
st.markdown("""
    <style>
    /* Dark charcoal steel background gradient */
    .stApp {
        background: linear-gradient(135deg, #090b0e 0%, #12161c 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* Premium High-End Instant Gradient Output Card */
    .premium-result-card {
        background: linear-gradient(135deg, #10141d 0%, #171e2c 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 15px 35px rgba(245, 158, 11, 0.15);
        margin: 20px auto;
        max-width: 850px;
        animation: cardGlowEntrance 0.6s ease-out forwards;
    }
    
    @keyframes cardGlowEntrance {
        0% { transform: translateY(10px); opacity: 0; box-shadow: 0 0 0 rgba(245, 158, 11, 0); }
        100% { transform: translateY(0); opacity: 1; box-shadow: 0 15px 35px rgba(245, 158, 11, 0.15); }
    }
    </style>
""", unsafe_allow_html=True)

# Main Structural Banner Layout
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 10px;">
        <h1 style="color: #f59e0b; font-family: sans-serif; font-weight: 900; letter-spacing: -0.5px; margin: 0;">⚡ CORE SPECIFICATION VALUATION</h1>
        <p style="color: #64748b; font-size: 1rem; margin-top: 5px;">Gradient Boosting Regressor Framework v2.4.0</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. SYSTEM FILE DISCOVERY & PIPELINE LOADER
# ==========================================
@st.cache_resource 
def load_assets():
    all_files = os.listdir(".")
    model_file = None
    for f in all_files:
        if f.startswith("laptop_price") and (f.endswith(".lib") or f.endswith(".pkl")):
            model_file = f
            break
            
    if model_file is None:
        raise FileNotFoundError("System Core Warning: Missing binary model matrix starting with 'laptop_price'.")

    model = joblib.load(model_file)
    le_company = joblib.load("company_encoder.lib")
    le_typename = joblib.load("type_encoder.lib")       
    le_screen = joblib.load("screen_resolution_encoder.lib")
    le_cpu = joblib.load("cpu_encoder.lib")
    le_gpu = joblib.load("gpu_encoder.lib")
    le_opsys = joblib.load("op_encoder.lib")            
    le_memory = joblib.load("memory_encoder.lib")        
    return model, le_company, le_typename, le_screen, le_cpu, le_gpu, le_opsys, le_memory

try:
    model, le_company, le_typename, le_screen, le_cpu, le_gpu, le_opsys, le_memory = load_assets()
except Exception as e:
    st.error(f"🚨 Pipeline Hardware Integration Interrupted: {e}")
    st.stop()

# ==========================================
# 3. INTERACTIVE HARDWARE FORM SELECTION
# ==========================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🏢 BRAND & FORM FACTOR", divider="orange")
    company = st.selectbox("Manufacturer Brand Line", le_company.classes_)
    typename = st.selectbox("Device Chassis Structure Target", le_typename.classes_)
    inches = st.slider("Display Boundary Size (Diagonal Inches)", 10.0, 18.0, 15.6, step=0.1)
    resolution = st.selectbox("Display Native Pixel Grid Layout", le_screen.classes_)

with col2:
    st.subheader("🧠 PERFORMANCE ARCHITECTURE", divider="orange")
    cpu = st.selectbox("Processing Engine Silicon (CPU)", le_cpu.classes_)
    gpu = st.selectbox("Graphics Engine Microchip (GPU)", le_gpu.classes_)
    ram = st.selectbox("Volatile System Cache Volume (RAM GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    memory = st.selectbox("Storage Block Partition Module (Drive Memory)", le_memory.classes_)
    opsys = st.selectbox("Platform Kernel Ecosystem (OpSys)", le_opsys.classes_)
    weight = st.slider("Chassis Mass Constraints (Weight in kg)", 0.5, 5.0, 1.8, step=0.1)

st.write("") # Structural Spacer

# ==========================================
# 4. PREDICTION RUNTIME & VISUAL CARD RENDERER
# ==========================================
if st.button("🔮 GENERATE PRICE VALUATION", type="primary", use_container_width=True):
    user_inputs = {
        'Company': company, 'TypeName': typename, 'Inches': inches,
        'ScreenResolution': resolution, 'Cpu': cpu, 'Ram': int(ram),
        'Memory': memory, 'Gpu': gpu, 'OpSys': opsys, 'Weight': weight
    }
    
    input_df = pd.DataFrame([user_inputs])
    
    input_df['Company'] = le_company.transform(input_df['Company'])
    input_df['TypeName'] = le_typename.transform(input_df['TypeName'])
    input_df['ScreenResolution'] = le_screen.transform(input_df['ScreenResolution'])
    input_df['Cpu'] = le_cpu.transform(input_df['Cpu'])
    input_df['Memory'] = le_memory.transform(input_df['Memory'])  
    input_df['Gpu'] = le_gpu.transform(input_df['Gpu'])
    input_df['OpSys'] = le_opsys.transform(input_df['OpSys'])
    
    feature_order = list(model.feature_names_in_)
    input_df = input_df[feature_order]
    
    prediction_array = model.predict(input_df)
    
    # 🌟 SOLUTION FIX: Unpack the number from the array using standard index slicing
    predicted_raw = float(prediction_array[0])
    
    # If your model targets a log scale index (np.log1p), uncomment the line below:
    # predicted_raw = float(np.expm1(predicted_raw))
    

    # Render the layout results card instantly
    st.markdown(f"""
        <div class="premium-result-card">
            <div style="color: #f59e0b; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 4px; font-weight: 700; font-family: monospace;">
                ⚡ AUTOMATED VALUATION ENGINE OUTPUT
            </div>
            <div style="color: #ffffff; font-size: 3.6rem; font-weight: 900; margin: 15px 0; letter-spacing: -1px; text-shadow: 0 0 20px rgba(245, 158, 11, 0.35);">
                {predicted_raw:,.2f} <span style="font-size: 1.6rem; font-weight: 300; color: #64748b; font-family: monospace;">Units</span>
            </div>
            <div style="border-top: 1px solid rgba(245, 158, 11, 0.15); padding-top: 14px; color: #64748b; font-size: 0.8rem; font-family: monospace; letter-spacing: 0.5px;">
                Core Architecture: <strong>Optimized Gradient Boosting Regressor</strong> | Node Status: <strong>Verified Online</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)
