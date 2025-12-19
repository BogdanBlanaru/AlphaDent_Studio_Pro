import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

class AlphaDentEngine:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing SAM 3 on {self.device}...")
        
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            self.mock_mode = False
        except Exception as e:
            print(f"⚠️ MODEL LOAD ERROR: {e}")
            self.mock_mode = True

        # --- BALANCED PROMPTS (Reliable) ---
        self.prompts = {
            # ANATOMY (Specific shapes to stop "Canine" invasion)
            9: "large wide square molar tooth back of mouth",  # Molar (Wide/Square)
            10: "round bicuspid premolar tooth",               # Premolar (Round)
            11: "sharp pointed canine tooth corner of mouth",  # Canine (Pointed)
            12: "flat rectangular incisor tooth front of mouth", # Incisor (Flat)

            # RESTORATIONS
            0: "scratched worn enamel surface",     
            1: "silver amalgam or white composite filling patch", 
            2: "unnatural gold or bright white artificial crown cap", # Specific for that yellow tooth
            
            # CARIES (Texture ONLY - Stop selecting shiny cusps)
            3: "dark black decay line in fissure",            
            4: "dark shadow stain spot between teeth",                      
            5: "brown decay spot on side",                    
            6: "black fracture line on edge",              
            7: "dark hole cavity near gum",               
            8: "black decay dot on tooth tip", # Changed from "decay spot" to "black dot" to ignore white reflections
        }

        self.labels = {
            0: "Abrasion", 1: "Filling", 2: "Crown", 
            3: "Caries (Fissure)", 4: "Caries (Molar Side)", 
            5: "Caries (Incisor Side)", 6: "Caries (Incisor Edge)", 
            7: "Caries (Gum)", 8: "Caries (Cusp)",
            9: "Molar", 10: "Premolar", 11: "Canine", 12: "Incisor"
        }

    def _calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _smart_nms(self, detections, width, height):
        """
        CALIBRATED FILTERING (v7)
        """
        if not detections: return []
        
        final = []
        image_area = width * height
        
        # 1. Sort by confidence
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        # Map to track claimed anatomy pixels
        anatomy_map = np.zeros((height, width), dtype=bool)

        for current in detections:
            mask = current['mask']
            mask_px = np.sum(mask)
            ratio = mask_px / image_area
            cid = current['id']

            # --- FILTER 1: SIZE LIMITS (Calibrated) ---
            
            # Caries (3-8): Limit to 5% (was 1.5%). 
            # 5% is small enough to stop "Whole Tooth" selection but big enough for large cavities.
            if 3 <= cid <= 8:
                if ratio > 0.05: continue 
            
            # Restorations (0-2): Limit to 30%.
            if cid in [0, 1, 2]:
                if ratio > 0.30: continue

            # Anatomy (9-12): Limit to 90% (was 35%).
            # This fixes the "Missing Teeth" bug in zoomed photos.
            if cid >= 9:
                if ratio > 0.90: continue

            # --- FILTER 2: ANATOMY LOGIC ---
            if cid >= 9:
                # If > 30% of this tooth is already claimed by a higher-confidence tooth, skip it.
                overlap = np.logical_and(mask, anatomy_map).sum()
                if (overlap / mask_px) > 0.30: continue
                
                anatomy_map = np.logical_or(anatomy_map, mask)
                final.append(current)
            
            # --- FILTER 3: PATHOLOGY LOGIC ---
            else:
                # Don't let pathologies overlap each other too much (30% max)
                is_dup = False
                for kept in final:
                    if kept['id'] < 9: 
                        iou = self._calculate_iou(mask, kept['mask'])
                        if iou > 0.3: 
                            is_dup = True
                            break
                if not is_dup:
                    final.append(current)

        return final

    def analyze(self, image_pil, conf_threshold=0.35): # Lowered default to catch more
        if self.mock_mode: return []

        raw_detections = []
        image_pil = image_pil.convert("RGB")
        width, height = image_pil.size
        
        print(f"   Processing {len(self.prompts)} prompts...")
        
        for class_id, prompt_text in self.prompts.items():
            try:
                inputs = self.processor(
                    images=image_pil, 
                    text=prompt_text, 
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs, 
                    threshold=conf_threshold, 
                    target_sizes=[(height, width)]
                )[0]

                for i, score in enumerate(results["scores"]):
                    mask = results["masks"][i].cpu().numpy().astype(bool)
                    
                    # Keep tiny specks out, but allow small cavities
                    if mask.sum() < 10: continue 
                    
                    raw_detections.append({
                        "id": class_id,
                        "label": self.labels[class_id],
                        "conf": float(score),
                        "mask": mask
                    })
            except Exception:
                continue
        
        print(f"   Raw Detections: {len(raw_detections)}")
        final = self._smart_nms(raw_detections, width, height)
        print(f"   Final Detections: {len(final)}")
        return final