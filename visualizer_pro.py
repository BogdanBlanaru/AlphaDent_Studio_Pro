import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    9: (255, 255, 255), 1: (128, 128, 128), 2: (70, 130, 180),
    3: (255, 50, 50), 4: (255, 100, 0), 5: (255, 200, 0),
    6: (150, 0, 150), 7: (200, 0, 50), 8: (255, 20, 100)
}

def get_tooth_name(fdi_number):
    try:
        n = int(fdi_number)
        digit = n % 10 # Last digit defines type (1=Central, 8=Wisdom)
        
        # 1-8 applies to ALL quadrants (1, 2, 3, 4)
        if digit in [1, 2]: return "Incisor"
        if digit == 3:      return "Canine"
        if digit in [4, 5]: return "Premolar"
        if digit in [6, 7, 8]: return "Molar"
    except: pass
    return "Tooth"

def draw_pro_overlay(image_pil, detections):
    img = np.array(image_pil)
    mask_layer = np.zeros_like(img)
    hud_layer = img.copy()
    
    sorted_dets = sorted(detections, key=lambda x: x['id'], reverse=False)
    
    for d in sorted_dets:
        if d['id'] == 9: continue 
        color = COLORS.get(d['id'], (255, 255, 255))
        mask_uint8 = d['mask'].astype(np.uint8)
        mask_layer[d['mask']] = color
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(hud_layer, contours, -1, color, 2)

    alpha_mask = np.any(mask_layer > 0, axis=-1)
    if np.any(alpha_mask):
        hud_layer[alpha_mask] = cv2.addWeighted(hud_layer[alpha_mask], 0.6, mask_layer[alpha_mask], 0.4, 0)
    
    final_pil = Image.fromarray(hud_layer)
    draw = ImageDraw.Draw(final_pil)
    
    try: font = ImageFont.truetype("arial.ttf", 18)
    except: font = ImageFont.load_default()
    
    for d in sorted_dets:
        ys, xs = np.where(d['mask'])
        if len(xs) == 0: continue
        x, y = int(np.mean(xs)), int(np.mean(ys))
        
        if d['id'] == 9: # TOOTH
            fdi = d.get('fdi', '')
            if fdi:
                # Cleaner Look: White text with shadow
                draw.text((x-11, y-11), fdi, fill=(0,0,0), font=font)
                draw.text((x-10, y-10), fdi, fill=(240,240,240), font=font)
        else: # PATHOLOGY
            text = d['label']
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill=(0,0,0, 160))
            draw.text((bbox[0], bbox[1]), text, fill=COLORS.get(d['id']), font=font)

    return final_pil

def generate_csv_report(detections, patient_id="Unknown"):
    data = []
    teeth_map = {d['fdi']: d for d in detections if d['id'] == 9 and 'fdi' in d}
    pathologies = [d for d in detections if d['id'] != 9]
    
    for fdi, tooth in teeth_map.items():
        status = "Healthy"
        action = "Monitor"
        conf = tooth['conf']
        
        t_mask = tooth['mask']
        for p in pathologies:
            p_mask = p['mask']
            if np.logical_and(t_mask, p_mask).any():
                status = p['label']
                conf = p['conf']
                if "Caries" in status: action = "Review"
                elif "Filling" in status: action = "Monitor (Restored)"
                break 
        
        data.append({
            "Patient ID": patient_id,
            "Tooth #": fdi,
            "Type": get_tooth_name(fdi),
            "Status": status,
            "Confidence": f"{conf:.1%}",
            "Action": action
        })

    df = pd.DataFrame(data)
    if not df.empty:
        # Sort properly (Numeric sort)
        df['sort_key'] = df['Tooth #'].astype(int)
        df = df.sort_values(by="sort_key").drop(columns=['sort_key'])
    return df