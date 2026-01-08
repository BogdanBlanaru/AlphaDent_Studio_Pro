import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from skimage import exposure

class AlphaDentEngine:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing AlphaDent Pro (v4 - Jaw Logic) on {self.device}...")
        
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            self.mock_mode = False
        except Exception:
            self.mock_mode = True

        # Medical Prompts
        self.prompts = {
            9: "white dental tooth structure enamel", 
            1: "silver grey amalgam dental filling restoration", 
            2: "artificial dental crown cap",
            3: "dark brown cavity fissure decay",            
            4: "dark shadow interproximal decay",                      
            5: "brown caries lesion spot",                    
            6: "black structural fracture crack",              
            7: "dark decay hole gumline",               
            8: "small black caries pit",                                 
        }

        self.labels = {
            0: "Abrasion", 1: "Filling (Amalgam)", 2: "Crown", 
            3: "Caries (Fissure)", 4: "Caries (Proximal)", 
            5: "Caries (Smooth)", 6: "Caries (Edge)", 
            7: "Caries (Cervical)", 8: "Caries (Cusp)",
            9: "Tooth" 
        }

    def _enhance_contrast(self, image_np):
        img_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

    def _analyze_material_physics(self, image_np, mask, class_id):
        # Physics verification for Amalgam vs Caries
        masked_px = image_np[mask]
        if masked_px.size == 0: return 0.0

        hsv_px = cv2.cvtColor(masked_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        avg_s = np.mean(hsv_px[:, 0, 1]) 
        avg_v = np.mean(hsv_px[:, 0, 2]) 
        
        if class_id == 1: # Amalgam
            if avg_s > 60: return 0.5 
            return 1.2

        if 3 <= class_id <= 8: # Caries
            if avg_v > 200 and avg_s < 20: return 0.0 # Glare check
            if avg_v < 100 and avg_s < 30: return 0.8 # Dark grey check
                
        return 1.0

    def analyze(self, image_pil, conf_threshold=0.30):
        if self.mock_mode: return []

        img_np = np.array(image_pil.convert("RGB"))
        enhanced_img = self._enhance_contrast(img_np)
        width, height = image_pil.size
        img_area = width * height
        
        raw_detections = []

        for class_id, prompt_text in self.prompts.items():
            try:
                inputs = self.processor(
                    images=Image.fromarray(enhanced_img), text=prompt_text, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs, threshold=conf_threshold, target_sizes=[(height, width)]
                )[0]

                for i, score in enumerate(results["scores"]):
                    mask = results["masks"][i].cpu().numpy().astype(bool)
                    
                    physics_score = self._analyze_material_physics(img_np, mask, class_id)
                    final_conf = float(score) * physics_score
                    
                    ratio = np.sum(mask) / img_area
                    if class_id < 9 and ratio > 0.15: continue 
                    if final_conf < conf_threshold: continue

                    raw_detections.append({
                        "id": class_id,
                        "label": self.labels[class_id],
                        "conf": final_conf,
                        "mask": mask
                    })
            except: continue

        return self._hierarchical_processing(raw_detections, width, height)

    def _hierarchical_processing(self, detections, width, height):
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        teeth = [d for d in detections if d['id'] == 9]
        pathologies = [d for d in detections if d['id'] != 9]
        
        # CLEAN UP TEETH
        final_teeth = []
        for t in teeth:
            is_dup = False
            for ft in final_teeth:
                intersection = np.logical_and(t['mask'], ft['mask']).sum()
                union = np.logical_or(t['mask'], ft['mask']).sum()
                if (intersection / union) > 0.30: is_dup = True; break
            if not is_dup: final_teeth.append(t)
            
        # --- NEW: DETECT JAW ORIENTATION ---
        is_upper_jaw = self._detect_jaw_type(final_teeth, height)
        print(f"DEBUG: Detected Jaw -> {'UPPER (Maxilla)' if is_upper_jaw else 'LOWER (Mandible)'}")

        # ASSIGN FDI NUMBERS
        final_teeth = self._assign_fdi_notation(final_teeth, width, is_upper_jaw)
        
        # PATHOLOGY MATCHING
        final_paths = []
        for p in pathologies:
            on_tooth = False
            for t in final_teeth:
                overlap = np.logical_and(p['mask'], t['mask']).sum()
                if (overlap / p['mask'].sum()) > 0.4: on_tooth = True; break
            if on_tooth: final_paths.append(p)
                
        return final_teeth + final_paths

    def _detect_jaw_type(self, teeth, height):
        """
        Returns True if Upper Jaw (Maxilla), False if Lower Jaw (Mandible).
        Logic: Checks Y-position of central teeth vs outer teeth.
        """
        if len(teeth) < 4: return True # Default to Upper if not enough info
        
        # Sort by X to find Left/Right/Center
        teeth.sort(key=lambda x: np.mean(np.where(x['mask'])[1]))
        
        # Extract centroids
        centroids = []
        for t in teeth:
            ys, xs = np.where(t['mask'])
            centroids.append(np.mean(ys))
            
        # Compare "Inner" (Incisors) vs "Outer" (Molars)
        # Take middle 25% as Incisors, outer 25% as Molars
        n = len(centroids)
        incisor_idx = slice(int(n*0.4), int(n*0.6) + 1)
        molar_idx_l = slice(0, int(n*0.2) + 1)
        molar_idx_r = slice(int(n*0.8), n)
        
        avg_incisor_y = np.mean(centroids[incisor_idx])
        avg_molar_y = (np.mean(centroids[molar_idx_l]) + np.mean(centroids[molar_idx_r])) / 2
        
        # GEOMETRY CHECK:
        # Standard Dental Photo:
        # If Incisors are HIGHER (Smaller Y) than Molars -> Upper Jaw (Arch points down)
        # If Incisors are LOWER (Larger Y) than Molars -> Lower Jaw (Arch points up)
        
        if avg_incisor_y > avg_molar_y:
            return False # MANDIBLE (Lower)
        else:
            return True # MAXILLA (Upper)

    def _assign_fdi_notation(self, teeth, width, is_upper_jaw):
        # Sort Left to Right (Image Coordinates)
        teeth.sort(key=lambda x: np.mean(np.where(x['mask'])[1]))
        center_line = width / 2
        
        # Split Image Left (Patient Right) / Image Right (Patient Left)
        img_left_teeth = [t for t in teeth if np.mean(np.where(t['mask'])[1]) < center_line] 
        img_right_teeth = [t for t in teeth if np.mean(np.where(t['mask'])[1]) >= center_line]
        
        # ASSIGN QUADRANTS
        if is_upper_jaw:
            # QUADRANT 1 (Patient Right / Img Left) -> 18 to 11
            img_left_teeth.reverse() 
            q1_code = 1
            # QUADRANT 2 (Patient Left / Img Right) -> 21 to 28
            q2_code = 2
        else:
            # LOWER JAW LOGIC
            # QUADRANT 4 (Patient Right / Img Left) -> 48 to 41
            # Wait! In FDI, Lower Right is Q4. Counting goes 41(center) -> 48(back).
            # Image Left is the back (molars) going to center? 
            # Usually: 
            # Img Left side = Patient Right side.
            # Outer edge = Molar (48), Center = Incisor (41).
            # So Img Left List is [48, 47 ... 41].
            # We need to reverse it to label correctly? No.
            # Let's count from Center Outwards.
            
            # We treat lists as moving from Center Outwards for simplicity
            img_left_teeth.reverse() # Now [Center ... Outer]
            q1_code = 4 # Lower Right
            
            # Image Right List is [Center ... Outer] naturally?
            # Img Right is [Center ... Outer] (21...28)
            q2_code = 3 # Lower Left
            
        # Apply Codes
        for i, t in enumerate(img_left_teeth):
            if i >= 8: t['id'] = -1; continue
            t['fdi'] = f"{q1_code}{i+1}"
            
        for i, t in enumerate(img_right_teeth):
            if i >= 8: t['id'] = -1; continue
            t['fdi'] = f"{q2_code}{i+1}"
            
        return [t for t in teeth if t.get('id') != -1]