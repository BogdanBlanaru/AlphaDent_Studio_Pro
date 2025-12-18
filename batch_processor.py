import os
import pandas as pd
from PIL import Image
import streamlit as st
from utils import mask_to_polygon
import time

def process_batch(model, input_folder, output_folder, conf_threshold=0.35, stop_event=None):
    """
    Iterates through all images in input_folder, runs SAM 3, 
    and saves YOLO-format .txt files to output_folder.
    """
    # 1. Setup
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    if total_images == 0:
        return 0, 0, "No images found in the selected folder."

    # 2. Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    start_time = time.time()
    
    # 3. Processing Loop
    for i, filename in enumerate(image_files):
        if stop_event and stop_event.is_set():
            status_text.warning("🛑 Batch processing stopped by user.")
            break
            
        file_path = os.path.join(input_folder, filename)
        file_id = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{file_id}.txt")
        
        # Update UI
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{total_images}: {filename}...")
        
        try:
            # A. Load Image
            image = Image.open(file_path).convert("RGB")
            
            # B. Run Inference (Using your refined SAM 3 Logic)
            detections = model.predict(image, conf_threshold=conf_threshold)
            
            # C. Format Results (YOLO Format: class x1 y1 ... xn yn)
            lines = []
            for det in detections:
                cid = det['class_id']
                # Pathologies only (0-8) are usually required for submission, 
                # but we can save anatomy (9-12) too if needed.
                # The prompt asks for 9 classes (0-8), so we filter.
                if cid > 8: continue 
                
                polygons = mask_to_polygon(det['mask'])
                
                for poly in polygons:
                    # Format: <class_id> <confidence> <poly_coords>
                    # Competition format: patient_id,class_id,confidence,poly
                    # But standard YOLO txt is: class_id x1 y1 ...
                    # Let's save a Submission CSV line format in a separate CSV later?
                    # For now, let's save the TXT annotations.
                    lines.append(f"{cid} {det['confidence']:.4f} {poly}")
            
            # D. Save to .txt
            with open(output_file, "w") as f:
                f.write("\n".join(lines))
                
            processed_count += 1
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

    # 4. Summary
    elapsed = time.time() - start_time
    status_text.success(f"✅ Completed! Processed {processed_count} images in {elapsed:.1f}s.")
    return processed_count, total_images, None