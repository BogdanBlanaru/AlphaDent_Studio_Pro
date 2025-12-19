import streamlit as st
import os
import pandas as pd
import time
import numpy as np 
from PIL import Image
from sam3_logic import AlphaDentEngine
from visualizer import draw_clinical_overlay, mask_to_yolo_poly

st.set_page_config(page_title="AlphaDent Pro", layout="wide", page_icon="🦷")
st.markdown("""<style>.stApp { background-color: #0E1117; } div.stButton > button { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: black; font-weight: bold; border: none;} h1, h2, h3 { color: #f0f2f6; } .stMetric { background-color: #1E1E1E; padding: 10px; border-radius: 8px; }</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_engine(): return AlphaDentEngine()

try: engine = load_engine()
except Exception as e: st.stop()

with st.sidebar:
    st.title("AlphaDent Pro")
    st.caption("v7.0 • Calibrated Logic")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35) 
    st.info("Tip: Lower confidence (0.25-0.35) catches more subtle caries.")

tab1, tab2 = st.tabs(["🩺 Diagnostics", "🏭 Batch"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        upload = st.file_uploader("Upload", type=["jpg", "png"])
        if upload:
            img = Image.open(upload).convert("RGB")
            st.image(img, width="stretch")
            if st.button("⚡ Run Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    start = time.time()
                    st.session_state['res'] = engine.analyze(img, conf_thresh)
                    st.session_state['img'] = img
                    st.session_state['time'] = time.time() - start

    with col2:
        if 'res' in st.session_state:
            res = st.session_state['res']
            
            # Always show the image, even if results are empty
            st.image(draw_clinical_overlay(st.session_state['img'], res), width="stretch")
            
            st.divider()
            c_path = sum(1 for x in res if x['id'] < 9)
            st.metric("Pathologies Detected", c_path, delta=f"{st.session_state['time']:.1f}s")
            
            if not res:
                st.warning("⚠️ No findings detected. Try lowering the 'Confidence Threshold' in the sidebar.")
            else:
                data = [{"Class": r['label'], "Conf": f"{r['conf']:.1%}", "Size": np.sum(r['mask'])} for r in res]
                st.dataframe(pd.DataFrame(data), use_container_width=True)

with tab2:
    st.header("Batch Processor")
    input_dir = st.text_input("Input Folder", "dataset/images/test")
    output_file = st.text_input("Output File", "submission.txt")
    if st.button("🚀 Start Batch"):
        if os.path.exists(input_dir):
            files = os.listdir(input_dir)
            bar = st.progress(0)
            lines = ["patient_id,class_id,confidence,poly"]
            for i, f in enumerate(files):
                if not f.endswith(('.jpg', '.png')): continue
                path = os.path.join(input_dir, f)
                img = Image.open(path).convert("RGB")
                dets = engine.analyze(img, conf_thresh)
                pid = os.path.splitext(f)[0]
                for d in dets:
                    for p in mask_to_yolo_poly(d['mask'], img.width, img.height):
                        lines.append(f'{pid},{d["id"]},{d["conf"]:.6f},"{p}"')
                bar.progress((i+1)/len(files))
            with open(output_file, "w") as f: f.write("\n".join(lines))
            st.success("Done!")
            st.download_button("Download", "\n".join(lines), "submission.txt")