import cv2
import numpy as np
import colorsys

def mask_to_polygon(binary_mask):
    """
    Converts a binary mask (0/1) to a normalized polygon string (x1 y1 x2 y2 ...).
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    height, width = binary_mask.shape
    
    for contour in contours:
        # Simplify contour to remove jagged edges (smooths the polygon)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Flatten array and normalize coordinates to 0-1 range
        points = approx.flatten().tolist()
        normalized_points = []
        
        for i in range(0, len(points), 2):
            x = points[i] / width
            y = points[i+1] / height
            # Format to 6 decimal places for precision
            normalized_points.append(f"{x:.6f}")
            normalized_points.append(f"{y:.6f}")
            
        # Filter out tiny noise (less than 3 points isn't a polygon)
        if len(normalized_points) >= 6: 
            polygons.append(" ".join(normalized_points))
            
    return polygons

def generate_distinct_colors(n):
    """Generates N distinct neon colors for the UI overlay."""
    HSV_tuples = [(x * 1.0 / n, 0.8, 0.9) for x in range(n)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

def overlay_masks(image, detections, classes):
    """
    Draws semi-transparent colored masks on the original image for the UI.
    """
    img_np = np.array(image)
    overlay = img_np.copy()
    
    colors = generate_distinct_colors(len(classes))
    
    # 1. Paint the masks on the overlay
    for det in detections:
        class_id = det['class_id']
        mask = det['mask']
        
        # Get color for this class (convert 0-1 RGB to 0-255)
        color = [int(c * 255) for c in colors[class_id]]
        
        # Draw the mask (color the pixels where mask is True)
        # Using [0,1,2] for RGB channels
        overlay[mask > 0] = color

    # 2. Blend original image and overlay (0.4 = 40% transparency)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0, img_np)
    
    return img_np