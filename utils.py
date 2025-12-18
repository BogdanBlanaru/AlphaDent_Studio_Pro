import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# --- PREMIUM DISTINCT COLOR PALETTE (R, G, B) ---
CLASS_COLORS = {
    # Pathologies
    0: (255, 50, 50),    # Red (Abrasion)
    1: (50, 255, 50),    # Lime Green (Filling)
    2: (50, 50, 255),    # Blue (Crown)
    3: (255, 255, 0),    # Yellow (Caries Fissure)
    4: (255, 165, 0),    # Orange (Caries Molar Side)
    5: (0, 255, 255),    # Cyan (Caries Incisor Side)
    6: (255, 0, 255),    # Magenta (Caries Incisor Edge)
    7: (128, 0, 128),    # Purple (Caries Gum Line)
    8: (255, 192, 203),  # Pink (Caries Cusp)
    
    # Anatomy
    9:  (220, 220, 220), # Light Gray (Molar)
    10: (180, 180, 180), # Medium Gray (Premolar)
    11: (255, 215, 0),   # Gold (Canine)
    12: (240, 230, 140)  # Khaki (Incisor)
}

# Short Labels for the Image
LABEL_MAP = {
    9: "M",   # Molar
    10: "P",  # Premolar
    11: "C",  # Canine
    12: "I"   # Incisor
}

def mask_to_polygon(mask_np):
    h, w = mask_np.shape
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.flatten().astype(float)
            contour[0::2] /= w
            contour[1::2] /= h
            poly_str = " ".join(f"{coord:.6f}" for coord in contour)
            polygons.append(poly_str)
    return polygons

def overlay_masks(image_pil, detections, class_prompts, alpha=0.35):
    """
    Refined Overlay with Abbreviated Labels.
    """
    image_np = np.array(image_pil)
    mask_layer = np.zeros_like(image_np)
    alpha_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=float)

    # 1. Colors
    for det in detections:
        cid = det['class_id']
        mask = det['mask']
        if not np.any(mask): continue
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        mask_layer[mask] = color
        alpha_mask[mask] = alpha

    # 2. Blending
    alpha_broadcast = alpha_mask[:, :, np.newaxis]
    blended = (image_np * (1 - alpha_broadcast) + mask_layer * alpha_broadcast).astype(np.uint8)
    
    # 3. Contours & Labels
    final_image = Image.fromarray(blended)
    draw = ImageDraw.Draw(final_image)
    
    try:
        # Slightly larger font for clarity but still unobtrusive
        font = ImageFont.truetype("arial.ttf", 13)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        cid = det['class_id']
        mask = det['mask']
        if not np.any(mask): continue

        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        
        # Draw thinner contour
        blended_with_lines = np.array(final_image)
        cv2.drawContours(blended_with_lines, contours, -1, color, 1)
        final_image = Image.fromarray(blended_with_lines)
        draw = ImageDraw.Draw(final_image)

        # Labels for Anatomy (Abbreviated to keep image clean)
        if cid in LABEL_MAP:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                label_text = LABEL_MAP[cid]
                # Tiny black box for readability
                bbox = draw.textbbox((cX, cY), label_text, font=font)
                draw.rectangle(bbox, fill="black")
                draw.text((cX, cY), label_text, fill="white", font=font, anchor="mm")

    return final_image