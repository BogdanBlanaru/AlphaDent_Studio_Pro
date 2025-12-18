import torch
import numpy as np
from PIL import Image
import sys
import cv2  # Global import

# Robust Import Logic
try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("❌ CRITICAL ERROR: Libraries missing. Run pip install transformers accelerate huggingface_hub")
    sys.exit(1)

class AlphaDentSAM3:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- MISSION 2: PATHOLOGIES ---
        self.pathology_classes = {
            0: "dental abrasion wear on enamel surface",
            1: "dental amalgam or composite filling restoration",
            2: "artificial dental crown cap restoration",
            3: "dark dental caries decay in tooth fissure",
            4: "interproximal dental caries decay on molar side",
            5: "interproximal dental caries decay on incisor side",
            6: "dental caries decay on incisor edge",
            7: "cervical dental caries decay near gum line",
            8: "dental caries decay on tooth cusp tip"
        }

        # --- MISSION 3: ANATOMY (BALANCED) ---
        # FIX: Added specific shape/location keywords to stop "Incisor" from taking over.
        self.anatomy_classes = {
            9: "wide large molar tooth back of mouth",      # Key: WIDE, BACK
            10: "round premolar tooth middle of mouth",     # Key: ROUND, MIDDLE
            11: "pointed canine tooth corner of mouth",     # Key: POINTED, CORNER
            12: "narrow rectangular incisor tooth front of mouth" # Key: NARROW, FRONT
        }

        self.default_classes = {**self.pathology_classes, **self.anatomy_classes}

        self.class_labels = {
            0: "Abrasion (Wear)",
            1: "Filling (Restoration)",
            2: "Crown (Cap)",
            3: "Caries - Fissure",
            4: "Caries - Molar Side",
            5: "Caries - Incisor Side",
            6: "Caries - Incisor Edge",
            7: "Caries - Gum Line",
            8: "Caries - Cusp Tip",
            9: "Molar",
            10: "Premolar",
            11: "Canine",
            12: "Incisor"
        }
        
        self.model = None
        self.processor = None
        self.mock_mode = False
        
        self.load_model(model_id)

    def load_model(self, model_id):
        print(f"🔄 Loading SAM 3 Engine ({model_id})...")
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            print("✅ SAM 3 Engine: ONLINE")
            self.mock_mode = False
        except Exception as e:
            print(f"⚠️ ACCESS ERROR: {e}")
            print("Switching to Simulation Mode.")
            self.mock_mode = True

    def _calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0: return 0.0
        return intersection / union

    def _apply_nms(self, detections, iou_threshold=0.3):
        if not detections: return []
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best['mask'], det['mask'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        return keep

    def predict(self, image_pil, custom_prompts=None, conf_threshold=0.25):
        raw_results = []
        active_prompts = self.default_classes.copy()
        if custom_prompts:
            active_prompts.update(custom_prompts)

        if self.mock_mode: return []

        try:
            print(f"🧠 SAM 3 Inference started...")
            total_pixels = image_pil.size[0] * image_pil.size[1]
            
            for class_id, prompt_text in active_prompts.items():
                
                inputs = self.processor(
                    images=image_pil, 
                    text=prompt_text, 
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                target_sizes = [image_pil.size[::-1]]
                processed_results = self.processor.post_process_instance_segmentation(
                    outputs, 
                    threshold=conf_threshold, 
                    mask_threshold=0.5, 
                    target_sizes=target_sizes
                )[0]

                for i, score in enumerate(processed_results["scores"]):
                    conf = float(score)
                    mask_np = processed_results["masks"][i].cpu().numpy().astype(bool)
                    
                    # --- FILTERS ---
                    mask_area = np.sum(mask_np)
                    coverage = mask_area / total_pixels
                    
                    # Relaxed Size Check: Allow big teeth, block tiny noise
                    if coverage > 0.85: continue 
                    if coverage < 0.0005: continue

                    # NOTE: Removed solidity check to prevent filtering out complex cavities

                    raw_results.append({
                        "class_id": class_id,
                        "prompt_used": prompt_text, 
                        "mask": mask_np,
                        "confidence": conf
                    })

            # --- PHASE 2: SORTING ---
            
            # 1. Anatomy (Teeth Types 9-12)
            anatomy_dets = [d for d in raw_results if d['class_id'] >= 9]
            # Aggressive NMS: Force 1 label per tooth
            final_anatomy = self._apply_nms(anatomy_dets, iou_threshold=0.25)
            
            # 2. Pathologies (0-8)
            pathology_dets = [d for d in raw_results if d['class_id'] < 9]
            # Standard NMS
            final_pathology = self._apply_nms(pathology_dets, iou_threshold=0.3)
            
            # Combine
            final_results = final_anatomy + final_pathology
            
            print(f"✅ Final Detection Count: {len(final_results)}")
            return final_results

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return []