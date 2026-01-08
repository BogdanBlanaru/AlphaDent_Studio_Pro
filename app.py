import streamlit as st
import pandas as pd
import time
import numpy as np
from PIL import Image
from sam3_enhanced import AlphaDentEngine
from visualizer_pro import draw_pro_overlay, generate_csv_report

# CONFIG
st.set_page_config(page_title="AlphaDent AI | Clinical Suite", layout="wide", page_icon="🦷")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; } 
    h1 { color: #00e5ff; font-family: 'Helvetica Neue'; }
    .stButton > button { background: linear-gradient(45deg, #00e5ff, #2979ff); border: none; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine(): return AlphaDentEngine()

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dental-braces.png", width=80)
    st.title("AlphaDent AI")
    st.caption("Meta SAM 3 • Clinical Edition")
    st.divider()
    conf_thresh = st.slider("Sensitivity (Confidence)", 0.1, 0.9, 0.35)
    mode = st.radio("Processing Mode", ["Clinical Analysis", "Batch Segmentation"])

engine = load_engine()

if mode == "Clinical Analysis":
    st.header("Intraoral Diagnostics")
    col_upload, col_view = st.columns([1, 3])
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload Intraoral Photo", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Original", use_container_width=True)
            
            if st.button("⚡ START SCAN", use_container_width=True):
                with st.spinner("Processing... Applying CLAHE & SAM 3 Inference..."):
                    start_time = time.time()
                    st.session_state['results'] = engine.analyze(img, conf_thresh)
                    st.session_state['img'] = img
                    st.session_state['proc_time'] = time.time() - start_time

    with col_view:
        if 'results' in st.session_state and st.session_state['results']:
            res = st.session_state['results']
            img = st.session_state['img']
            
            tab_vis, tab_data = st.tabs(["👁️ Vision Overlay", "📋 Clinical Report"])
            
            with tab_vis:
                overlay_img = draw_pro_overlay(img, res)
                c1, c2 = st.columns(2)
                with c1: 
                    st.subheader("Enhanced AI View")
                    st.image(overlay_img, use_container_width=True)
                with c2: 
                    st.subheader("Contrast Enhanced Input")
                    st.image(engine._enhance_contrast(np.array(img)), use_container_width=True)

            with tab_data:
                teeth_count = sum(1 for x in res if x['id'] == 9)
                pathology_count = sum(1 for x in res if x['id'] < 9)
                m1, m2, m3 = st.columns(3)
                m1.metric("Teeth Identified", teeth_count)
                m2.metric("Pathologies Found", pathology_count, delta_color="inverse")
                m3.metric("Processing Time", f"{st.session_state['proc_time']:.2f}s")
                
                st.divider()
                df = generate_csv_report(res, uploaded_file.name)
                st.dataframe(df, use_container_width=True)

        elif 'results' in st.session_state:
            st.warning("No structures detected. Try adjusting contrast or sensitivity.")

else:
    st.header("Batch Processor")
    st.write("Batch mode active.")