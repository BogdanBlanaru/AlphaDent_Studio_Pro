import streamlit as st
from PIL import Image
import os
import pandas as pd
from sam3_handler import AlphaDentSAM3
from utils import mask_to_polygon, overlay_masks, CLASS_COLORS
import batch_processor 
import threading

# --- PRO PAGE CONFIG ---
st.set_page_config(
    page_title="AlphaDent Studio Pro",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #4facfe; }
    h3 { color: #00f2fe; }
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: bold;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); border: none; color: white;
    }
    .color-box {
        width: 12px; height: 12px; display: inline-block; margin-right: 5px; border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("AlphaDent Pro")
    st.caption("v3.5 • Batch Processing Active")
    
    @st.cache_resource
    def get_model():
        return AlphaDentSAM3("facebook/sam3")

    model = get_model()
    
    if model.mock_mode: st.warning("⚠️ SIMULATION MODE")
    else: st.success("🟢 SAM 3 ENGINE ACTIVE")
        
    st.markdown("---")
    global_conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

# --- MAIN APP ---
st.title("🦷 AlphaDent AI Studio")
tab1, tab2, tab3 = st.tabs(["🖼️ Mission Dashboard", "🧪 Prompt Lab", "📂 Batch Processor"])

# === TAB 1: DASHBOARD ===
with tab1:
    uploaded_file = st.file_uploader("Upload Intraoral Photo", type=['jpg', 'png', 'jpeg'], key="main_upload")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.container():
            st.subheader("Live Analysis")
            if st.button("⚡ Run Full Scan", type="primary"):
                with st.spinner('Running SAM 3 Vision Transformer...'):
                    st.session_state['detections'] = model.predict(image, conf_threshold=global_conf)
        
        if 'detections' in st.session_state and st.session_state['detections']:
            dets = st.session_state['detections']
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original Source")
                st.image(image, use_container_width=True)
            with c2:
                st.caption(f"SAM 3 Segmentation ({len(dets)} detections)")
                result_img = overlay_masks(image, dets, model.default_classes)
                st.image(result_img, use_container_width=True)
            
            with st.expander("🎨 Legend", expanded=True):
                st.markdown("**Anatomy (Teeth)**")
                cols = st.columns(4)
                for i in range(9, 13):
                    with cols[i-9]:
                        color_hex = '#%02x%02x%02x' % CLASS_COLORS[i]
                        st.markdown(f'<span class="color-box" style="background-color: {color_hex};"></span> {model.class_labels[i]}', unsafe_allow_html=True)
                
                st.markdown("**Pathologies**")
                cols = st.columns(5)
                for i in range(0, 9):
                    with cols[i%5]:
                        color_hex = '#%02x%02x%02x' % CLASS_COLORS[i]
                        st.markdown(f'<span class="color-box" style="background-color: {color_hex};"></span> {model.class_labels[i]}', unsafe_allow_html=True)

            st.markdown("### 📊 Clinical Findings")
            
            anatomy_data = []
            pathology_data = []
            submission_lines = []
            file_id = os.path.splitext(uploaded_file.name)[0]
            
            for d in dets:
                cid = d['class_id']
                label = model.class_labels.get(cid, "Unknown")
                item = {"ID": cid, "Description": label, "Confidence": d['confidence'], "Prompt Used": d['prompt_used']}
                if cid >= 9: anatomy_data.append(item)
                else: pathology_data.append(item)
                for p in mask_to_polygon(d['mask']):
                    submission_lines.append(f"{file_id},{cid},{d['confidence']:.4f},\"{p}\"")
            
            col_anat, col_path = st.columns(2)
            with col_anat:
                st.markdown("#### 🦷 Detected Anatomy")
                if anatomy_data: st.dataframe(pd.DataFrame(anatomy_data).style.format({"Confidence": "{:.2%}"}), use_container_width=True)
                else: st.info("No specific tooth anatomy detected.")

            with col_path:
                st.markdown("#### 🩺 Detected Pathologies")
                if pathology_data: st.dataframe(pd.DataFrame(pathology_data).style.format({"Confidence": "{:.2%}"}), use_container_width=True)
                else: st.success("No visible pathologies detected.")
            
            st.download_button("📥 Download Submission File", data="\n".join(submission_lines), file_name=f"{file_id}.txt")

# === TAB 2: PROMPT LAB ===
with tab2:
    st.header("🧪 Prompt Engineering")
    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        custom_prompts = {}
        with st.form("prompt_form"):
            for cid in range(0, 13):
                default = model.default_classes[cid]
                label = model.class_labels[cid]
                new_val = st.text_input(f"{label}", value=default)
                if new_val != default: custom_prompts[cid] = new_val
            run_test = st.form_submit_button("🧪 Run Test")

    with col_r:
        if uploaded_file and run_test:
            with st.spinner('Testing prompts...'):
                lab_dets = model.predict(image, custom_prompts=custom_prompts, conf_threshold=global_conf)
                lab_img = overlay_masks(image, lab_dets, model.default_classes)
                st.image(lab_img, use_container_width=True)
                st.success(f"Found {len(lab_dets)} objects.")

# === TAB 3: BATCH PROCESSOR (UPDATED) ===
with tab3:
    st.header("📂 Batch Processor")
    st.markdown("""
    Use this module to process the **Entire Kaggle Dataset** automatically.
    1.  Place your dataset images in a folder named `dataset/images/test`.
    2.  Click **Start Batch** below.
    3.  Results will be saved in `dataset/labels_pred`.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        input_dir = st.text_input("Input Image Folder", value="AlphaDent/images/test")
    with col2:
        output_dir = st.text_input("Output Label Folder", value="AlphaDent/labels_pred")
    
    if st.button("🚀 Start Batch Processing", type="primary"):
        if not os.path.exists(input_dir):
            st.error(f"Folder '{input_dir}' not found! Please create it and add images.")
        else:
            with st.spinner("Initializing Batch Pipeline..."):
                count, total, err = batch_processor.process_batch(model, input_dir, output_dir, conf_threshold=global_conf)
                if err:
                    st.error(err)
                else:
                    st.balloons()
                    st.success(f"✅ Batch Complete! Processed {count}/{total} images.")
                    st.info(f"Results saved to: `{output_dir}`")