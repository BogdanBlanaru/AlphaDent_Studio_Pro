import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Professional Medical Colors (High Contrast)
COLORS = {
    9: (240, 240, 245), # Enamel White (Subtle)
    1: (169, 169, 169), # Amalgam Silver
    2: (70, 130, 180),  # Crown Blue
    3: (255, 69, 0),    # Fissure Red (Alert)
    4: (255, 140, 0),   # Proximal Orange
    5: (255, 215, 0),   # Smooth Yellow
    6: (139, 0, 139),   # Edge Purple
    7: (220, 20, 60),   # Cervical Crimson
    8: (255, 20, 147)   # Cusp Pink
}

def get_tooth_name(fdi_number):
    """Maps FDI Code (e.g., 18) to Name (e.g., Molar)"""
    try:
        n = int(fdi_number)
        digit = n % 10
        if digit in [1, 2]: return "Incisor"
        if digit == 3:      return "Canine"
        if digit in [4, 5]: return "Premolar"
        if digit in [6, 7, 8]: return "Molar"
    except:
        pass
    return "Tooth"

def draw_pro_overlay(image_pil, detections):
    img = np.array(image_pil)
    # Create distinct layers for medical look
    mask_layer = np.zeros_like(img)
    hud_layer = img.copy()
    
    # 1. Draw Masks with smooth contours
    for d in detections:
        color = COLORS.get(d['id'], (255, 255, 255))
        mask = d['mask'].astype(np.uint8)
        
        # Fill mask layer
        mask_layer[d['mask']] = color
        
        # Smooth Contours (Medical Look)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Smoothing factor (epsilon)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(hud_layer, [approx], -1, color, 2, cv2.LINE_AA)

    # 2. Blend: Make teeth subtle (20%), Pathologies popping (50%)
    alpha_mask = np.any(mask_layer > 0, axis=-1)
    hud_layer[alpha_mask] = cv2.addWeighted(hud_layer[alpha_mask], 0.75, mask_layer[alpha_mask], 0.25, 0)
    
    final_pil = Image.fromarray(hud_layer)
    draw = ImageDraw.Draw(final_pil)
    
    # Load Fonts
    try: 
        font_lg = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except: 
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()
    
    # 3. Intelligent Labeling
    for d in detections:
        ys, xs = np.where(d['mask'])
        if len(xs) == 0: continue
        
        # Centroid
        x, y = int(np.mean(xs)), int(np.mean(ys))
        
        if d['id'] == 9: # ANATOMY (Teeth)
            # Show "18 Molar"
            fdi = d.get('fdi', '')
            name = get_tooth_name(fdi)
            label = f"{fdi}\n{name}" if fdi else "Tooth"
            
            # Draw Centered Gray Text
            w = draw.textlength(fdi, font=font_lg)
            draw.text((x - w/2, y - 10), fdi, fill=(255,255,255, 180), font=font_lg)
            
            # Small name below
            w2 = draw.textlength(name, font=font_sm)
            draw.text((x - w2/2, y + 15), name, fill=(200,200,200, 160), font=font_sm)
            
        else: # PATHOLOGY
            # High Visibility Badge
            label = d['label']
            conf_txt = f"{int(d['conf']*100)}%"
            
            # Offset pathology labels slightly up to avoid overlapping tooth center
            text = f"{label} ({conf_txt})"
            bbox = draw.textbbox((x, y-40), text, font=font_lg)
            
            # Black Box with Color Border
            draw.rectangle(bbox, fill=(0,0,0, 220), outline=COLORS.get(d['id']))
            draw.text((bbox[0], bbox[1]), text, fill=(255,255,255), font=font_lg)

    return final_pil

def generate_csv_report(detections, patient_id="Unknown"):
    data = []
    
    # 1. Map Teeth
    teeth_map = {d['fdi']: d for d in detections if d['id'] == 9 and 'fdi' in d}
    
    # 2. Add Pathologies
    pathologies = [d for d in detections if d['id'] != 9]
    for p in pathologies:
        # Link to nearest tooth
        p_mask = p['mask']
        ys, xs = np.where(p_mask)
        if len(xs) == 0: continue
        cx, cy = np.mean(xs), np.mean(ys)
        
        associated_tooth_num = "General"
        associated_tooth_type = "Unknown"
        min_dist = 10000
        
        for fdi, tooth in teeth_map.items():
            t_ys, t_xs = np.where(tooth['mask'])
            t_cx, t_cy = np.mean(t_xs), np.mean(t_ys)
            dist = np.sqrt((cx-t_cx)**2 + (cy-t_cy)**2)
            if dist < min_dist:
                min_dist = dist
                associated_tooth_num = fdi
                associated_tooth_type = get_tooth_name(fdi)
        
        data.append({
            "Patient ID": patient_id,
            "Tooth #": associated_tooth_num,
            "Type": associated_tooth_type,
            "Status": p['label'],
            "Confidence": f"{p['conf']:.1%}",
            "Action": "Treat"
        })

    # 3. Add Healthy Teeth (So you see the full count)
    for fdi, tooth in teeth_map.items():
        # Check if this tooth already has a pathology listed above
        has_pathology = any(x['Tooth #'] == fdi for x in data)
        if not has_pathology:
            data.append({
                "Patient ID": patient_id,
                "Tooth #": fdi,
                "Type": get_tooth_name(fdi),
                "Status": "Healthy",
                "Confidence": f"{tooth['conf']:.1%}",
                "Action": "Monitor"
            })
            
    # Sort by Tooth Number for clean table
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Tooth #")
        
    return df