import streamlit as st
from PIL import Image
import os
import pandas as pd
from sam3_handler import AlphaDentSAM3
from utils import mask_to_polygon, overlay_masks

# --- PRO PAGE CONFIG ---
st.set_page_config(
    page_title="AlphaDent Studio Pro",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (The "Premium" Look) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #4facfe; }
    h3 { color: #00f2fe; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("AlphaDent Pro")
    st.caption("v2.5 • Powered by SAM 3")
    
    st.markdown("---")
    
    # Model Status
    @st.cache_resource
    def get_model():
        return AlphaDentSAM3("facebook/sam3")

    model = get_model()
    
    if model.mock_mode:
        st.warning("⚠️ SIMULATION MODE")
    else:
        st.success("🟢 SAM 3 ENGINE ACTIVE")
        
    st.markdown("---")
    global_conf = st.slider("Global Confidence", 0.0, 1.0, 0.35, 0.05)

# --- MAIN UI ---
st.title("🦷 AlphaDent AI Studio")

# TABS for Professional Workflow
tab1, tab2, tab3 = st.tabs(["🖼️ Analysis Dashboard", "🧪 Prompt Lab (Fine-Tuning)", "📂 Batch Export"])

# === TAB 1: STANDARD DASHBOARD ===
with tab1:
    uploaded_file = st.file_uploader("Upload Intraoral Photo", type=['jpg', 'png', 'jpeg'], key="main_upload")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # 1. Top Bar: Original
        with st.container():
            st.subheader("Live Analysis")
            if st.button("⚡ Run SAM 3 Segmentation", type="primary"):
                with st.spinner('Scanning dental structures...'):
                    # Save detections to session state to prevent reload loss
                    st.session_state['detections'] = model.predict(image, conf_threshold=global_conf)
        
        # 2. Results View
        if 'detections' in st.session_state and st.session_state['detections']:
            dets = st.session_state['detections']
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Original Source", use_container_width=True)
            with c2:
                # Generate overlay
                result_img = overlay_masks(image, dets, model.default_classes)
                st.image(result_img, caption="SAM 3 Segmentation Overlay", use_container_width=True)
            
            # Metrics
            st.markdown("### 📊 Pathology Report")
            
            # Convert to DataFrame for nice display
            report_data = []
            submission_lines = []
            file_id = os.path.splitext(uploaded_file.name)[0]
            
            for d in dets:
                poly_count = len(mask_to_polygon(d['mask']))
                report_data.append({
                    "Class ID": d['class_id'],
                    "Pathology Name": model.default_classes[d['class_id']],
                    "Confidence": d['confidence'],
                    "Polygons": poly_count
                })
                # Generate CSV lines hiddenly
                for poly in mask_to_polygon(d['mask']):
                    submission_lines.append(f"{file_id},{d['class_id']},{d['confidence']:.4f},\"{poly}\"")
            
            df = pd.DataFrame(report_data)
            if not df.empty:
                st.dataframe(
                    df.style.background_gradient(subset=['Confidence'], cmap="Greens"),
                    use_container_width=True
                )
                
                # Export
                st.download_button(
                    "📥 Download Competition Submission File",
                    data="\n".join(submission_lines),
                    file_name=f"{file_id}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No pathologies found above confidence threshold.")

# === TAB 2: PROMPT LAB (THE WINNING FEATURE) ===
with tab2:
    st.header("🧪 Zero-Shot Prompt Engineering")
    st.info("💡 Pro Tip: If SAM 3 misses a cavity, try describing it visually (e.g., change 'cavity' to 'dark spot on enamel').")
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("Fine-Tune Class Definitions")
        custom_prompts = {}
        
        # Generate inputs for all classes
        for cid, default_text in model.default_classes.items():
            new_text = st.text_input(f"Class {cid} Prompt", value=default_text)
            if new_text != default_text:
                custom_prompts[cid] = new_text
        
        if custom_prompts:
            st.success(f"Applying {len(custom_prompts)} custom definitions.")

    with col_r:
        st.subheader("Test Area")
        if uploaded_file:
            if st.button("🧪 Test Custom Prompts"):
                with st.spinner('Running SAM 3 with experimental prompts...'):
                    # Run prediction with OVERRIDES
                    lab_dets = model.predict(image, custom_prompts=custom_prompts, conf_threshold=global_conf)
                    
                    # Visualize
                    lab_img = overlay_masks(image, lab_dets, 
                                          {**model.default_classes, **custom_prompts})
                    
                    st.image(lab_img, caption="Experimental Result", use_container_width=True)
                    
                    # Diff check
                    st.write(f"Detected {len(lab_dets)} objects with new prompts.")
        else:
            st.warning("Upload an image in the Dashboard tab first.")

# === TAB 3: BATCH EXPORT ===
with tab3:
    st.header("📂 Batch Processing")
    st.caption("Upload multiple files to generate a single submission.csv")
    # (Placeholder for future expansion)
    st.info("Batch processing module ready for implementation.")