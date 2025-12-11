import torch
import numpy as np
from PIL import Image
import sys
import time

# Robust Import Logic
try:
    from transformers import Sam3Processor, Sam3Model
    from huggingface_hub.utils import GatedRepoError
except ImportError:
    print("❌ CRITICAL ERROR: Libraries missing.")
    sys.exit(1)

class AlphaDentSAM3:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Default Competition Classes (Baseline)
        self.default_classes = {
            0: "abrasion on tooth surface",
            1: "dental filling material",
            2: "dental crown artificial tooth",
            3: "cavity in tooth fissure",
            4: "cavity on molar side",
            5: "cavity on incisor side",
            6: "cavity on incisor edge",
            7: "cavity near gum line",
            8: "cavity on tooth cusp tip"
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

    def predict(self, image_pil, custom_prompts=None, conf_threshold=0.30):
        """
        Runs prediction. 
        'custom_prompts' allows overriding specific class descriptions.
        """
        results = []
        
        # Use defaults or overrides
        active_classes = self.default_classes.copy()
        if custom_prompts:
            active_classes.update(custom_prompts)

        if self.mock_mode:
            # Simulation for UI testing
            import cv2
            w, h = image_pil.size
            mask = np.zeros((h, w), dtype=bool)
            cv2.circle(mask.view(np.uint8), (int(w/2), int(h/2)), 100, 1, -1)
            results.append({"class_id": 1, "mask": mask, "confidence": 0.95})
            return results

        try:
            print(f"🧠 SAM 3 Inference started...")
            
            # --- STABLE ITERATIVE INFERENCE ---
            # We process one class prompt at a time to ensure stability
            
            for class_id, prompt_text in active_classes.items():
                
                # Prepare inputs for this specific concept
                inputs = self.processor(
                    images=image_pil,
                    text=prompt_text,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                target_sizes = [image_pil.size[::-1]]
                
                # Post-process
                processed_results = self.processor.post_process_instance_segmentation(
                    outputs, 
                    threshold=conf_threshold, 
                    mask_threshold=0.5, 
                    target_sizes=target_sizes
                )[0]

                # Extract detections
                for i, score in enumerate(processed_results["scores"]):
                    conf = float(score)
                    
                    # Get mask
                    mask_tensor = processed_results["masks"][i]
                    mask_np = mask_tensor.cpu().numpy().astype(bool)
                    
                    # Store result
                    results.append({
                        "class_id": class_id,
                        "class_name": prompt_text,
                        "mask": mask_np,
                        "confidence": conf
                    })
                    print(f"   Found Class {class_id} ('{prompt_text}'): {conf:.2%}")

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            import traceback
            traceback.print_exc()

        return results