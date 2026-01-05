import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from skimage import exposure

class AlphaDentEngine:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing AlphaDent Pro on {self.device}...")
        
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            self.mock_mode = False
        except Exception:
            self.mock_mode = True

        # Enhanced Medical Prompts
        self.prompts = {
            9: "white dental tooth structure enamel", 
            1: "silver amalgam dental filling material", 
            2: "artificial dental crown restoration",
            3: "dark cavity fissure decay line",            
            4: "dark interproximal shadow cavity",                      
            5: "brown caries lesion spot on smooth surface",                    
            6: "black structural fracture crack",              
            7: "dark cervical decay hole gumline",               
            8: "small black caries pit",                                 
        }

        self.labels = {
            0: "Abrasion", 1: "Filling", 2: "Crown", 
            3: "Caries (Fissure)", 4: "Caries (Proximal)", 
            5: "Caries (Smooth)", 6: "Caries (Edge)", 
            7: "Caries (Cervical)", 8: "Caries (Cusp)",
            9: "Tooth" 
        }

    def _enhance_contrast(self, image_np):
        # CLAHE for revealing hidden decay
        img_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

    def analyze(self, image_pil, conf_threshold=0.30):
        if self.mock_mode: return []

        # 1. High-Res Pre-processing
        img_np = np.array(image_pil.convert("RGB"))
        enhanced_img = self._enhance_contrast(img_np)
        width, height = image_pil.size
        img_area = width * height
        
        raw_detections = []

        # 2. Inference Loop
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
                    
                    # --- PHYSICS & SIZE CHECK ---
                    mask_px = np.sum(mask)
                    ratio = mask_px / img_area
                    
                    # Constraint: Pathology cannot be >15% of image (Too big = Hallucination)
                    if class_id < 9 and ratio > 0.15: continue
                    # Constraint: Pathology must be somewhat dark (unless it's a white spot lesion, but sticking to basics)
                    if class_id < 9 and class_id != 2: # Exclude crowns
                        masked_px = img_np[mask]
                        if np.mean(masked_px) > 180: continue # Too bright to be decay
                    
                    raw_detections.append({
                        "id": class_id,
                        "label": self.labels[class_id],
                        "conf": float(score),
                        "mask": mask
                    })
            except: continue

        return self._hierarchical_processing(raw_detections, width)

    def _hierarchical_processing(self, detections, width):
        # Sort by confidence
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        # Split Anatomy / Pathology
        teeth = [d for d in detections if d['id'] == 9]
        pathologies = [d for d in detections if d['id'] != 9]
        
        # NMS for Teeth (Overlap removal)
        final_teeth = []
        for t in teeth:
            overlap = False
            for ft in final_teeth:
                iou = np.logical_and(t['mask'], ft['mask']).sum() / np.logical_or(t['mask'], ft['mask']).sum()
                if iou > 0.4: overlap = True; break
            if not overlap: final_teeth.append(t)
            
        # Assign FDI Numbers (Smart Split)
        final_teeth = self._assign_fdi_notation(final_teeth, width)
        
        # Filter Pathologies (Must be ON a tooth)
        final_paths = []
        for p in pathologies:
            on_tooth = False
            for t in final_teeth:
                intersection = np.logical_and(p['mask'], t['mask']).sum()
                if (intersection / p['mask'].sum()) > 0.5: # 50% inside a tooth
                     on_tooth = True; break
            
            # Simple NMS for pathologies
            if on_tooth:
                dup = False
                for fp in final_paths:
                    iou = np.logical_and(p['mask'], fp['mask']).sum() / np.logical_or(p['mask'], fp['mask']).sum()
                    if iou > 0.3: dup = True; break
                if not dup: final_paths.append(p)
                
        return final_teeth + final_paths

    def _assign_fdi_notation(self, teeth, width):
        # 1. Sort all teeth Left -> Right (Image coordinates)
        # Note: In dental photos, Left side of Image = Right side of Patient (usually)
        teeth.sort(key=lambda x: np.mean(np.where(x['mask'])[1]))
        
        # 2. Find the "Central Gap" (Largest X-distance change near center?)
        # Heuristic: The split is near width/2.
        center_line = width / 2
        
        # Split into Right Quadrant (Image Left) and Left Quadrant (Image Right)
        # We assume standard intraoral view
        q1_teeth = [t for t in teeth if np.mean(np.where(t['mask'])[1]) < center_line] # Patient Right
        q2_teeth = [t for t in teeth if np.mean(np.where(t['mask'])[1]) >= center_line] # Patient Left
        
        # 3. Assign IDs
        # Q1 (Patient Right): Count from Center (Rightmost in list) -> Outwards (Leftmost in list)
        # q1_teeth is [Molar...Incisor]. We need to reverse it to label 11, 12, 13...
        for i, t in enumerate(reversed(q1_teeth)):
            t['fdi'] = f"1{i+1}"
        
        # Q2 (Patient Left): Count from Center (Leftmost in list) -> Outwards
        # q2_teeth is [Incisor...Molar]. Correct order for 21, 22, 23...
        for i, t in enumerate(q2_teeth):
            t['fdi'] = f"2{i+1}"
            
        return teeth