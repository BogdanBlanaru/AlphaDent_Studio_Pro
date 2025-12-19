import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Dental Color Palette (High Contrast)
COLORS = {
    # Anatomy (Grays/Whites)
    9: (200, 200, 200), 10: (170, 170, 170), 11: (255, 215, 0), 12: (240, 240, 220),
    # Pathology (Hot Colors)
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 127, 80), 5: (0, 255, 255), 6: (255, 0, 255), 
    7: (128, 0, 128), 8: (255, 105, 180)
}

def mask_to_yolo_poly(mask, width, height):
    """Converts boolean mask to normalized flat polygon string"""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polys = []
    for cnt in contours:
        if len(cnt) > 2:
            # Flatten and normalize
            flat = cnt.flatten().astype(float)
            flat[0::2] /= width  # X coords
            flat[1::2] /= height # Y coords
            # Format: x1 y1 x2 y2 ...
            poly_str = " ".join([f"{x:.5f}" for x in flat])
            polys.append(poly_str)
    return polys

def draw_clinical_overlay(image_pil, detections):
    """Professional medical overlay"""
    img = np.array(image_pil)
    overlay = img.copy()
    
    # 1. Draw Masks
    for d in detections:
        color = COLORS.get(d['id'], (255, 255, 255))
        mask = d['mask']
        # Alpha blend
        overlay[mask] = (np.array(color) * 0.6 + overlay[mask] * 0.4).astype(np.uint8)
        
        # Contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    # 2. Add Labels (with logic to avoid clutter)
    final_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(final_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for d in detections:
        # Only label pathologies to keep it clean, or high-conf anatomy
        if d['id'] < 9 or (d['id'] >= 9 and d['conf'] > 0.85):
            mask = d['mask']
            ys, xs = np.where(mask)
            if len(xs) > 0:
                y_mid, x_mid = int(np.mean(ys)), int(np.mean(xs))
                text = f"{d['label']} {int(d['conf']*100)}%"
                
                # Text Background
                bbox = draw.textbbox((x_mid, y_mid), text, font=font)
                draw.rectangle(bbox, fill=(0,0,0, 180))
                draw.text((x_mid, y_mid), text, font=font, fill=COLORS.get(d['id']))
                
    return final_pil