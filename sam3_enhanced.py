import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from skimage import exposure

class AlphaDentEngine:
    def __init__(self, model_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing AlphaDent Pro (v5 - Physics Auto-Correct) on {self.device}...")
        
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(self.device)
            self.mock_mode = False
        except Exception:
            self.mock_mode = True

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
            0: "Abrasion", 1: "Filling (Amalgam)", 2: "Crown (Restoration)", 
            3: "Caries (Fissure)", 4: "Caries (Proximal)", 
            5: "Caries (Smooth)", 6: "Caries (Edge)", 
            7: "Caries (Cervical)", 8: "Caries (Cusp)",
            9: "Tooth" 
        }

    def _enhance_contrast(self, image_np):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) # Redus putin clipLimit pt a evita reflexiile false
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

    def _autocorrect_class(self, image_np, mask, original_class_id):
        """
        ANALIZA SPECTRALA A PIXELILOR:
        1. Reflexie (Glare) -> Sterge detectia.
        2. Amalgam (Metal) vs Carie (Organica) -> Corecteaza clasa.
        """
        masked_px = image_np[mask]
        if masked_px.size == 0: return original_class_id, 0.0

        # Convertim la HSV
        hsv_px = cv2.cvtColor(masked_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        
        avg_h = np.mean(hsv_px[:, 0, 0]) # Hue (Culoare)
        avg_s = np.mean(hsv_px[:, 0, 1]) # Saturation (Intensitate culoare)
        avg_v = np.mean(hsv_px[:, 0, 2]) # Value (Luminozitate)
        
        # --- FILTRUL 1: REFLEXII (GLARE KILLER) ---
        # Problema cu Incisivul 11: Reflexie alba puternica
        # Daca zona e FOARTE luminoasa (V > 210) si saturatie mica (S < 15) -> E lumina, nu carie.
        if 3 <= original_class_id <= 8:
            if avg_v > 200 and avg_s < 20: 
                return -1, 0.0 # -1 inseamna "Sterge detectia"

        # --- FILTRUL 2: AMALGAM VS CARIE ---
        # Problema Molarului 16: Amalgamul e GRI (Saturatie mica), Caria e MARO/ROSU (Saturatie medie)
        
        # Cazul: AI a zis "Carie", dar materialul e GRI METALIC
        if 3 <= original_class_id <= 8:
            # Daca e intunecat (V < 100) DAR fara culoare (S < 35) -> E Metal (Amalgam)
            if avg_v < 120 and avg_s < 40:
                return 1, 1.0 # FORCE SWITCH la Class 1 (Filling)
        
        # Cazul: AI a zis "Amalgam", dar e prea colorat (ex: gingie sau carie rosiatica)
        if original_class_id == 1:
            if avg_s > 60: # Prea colorat pentru argint
                return 5, 0.8 # Switch la Carie Generica (Smooth) cu conf mai mica

        return original_class_id, 1.0

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
                    
                    # 1. Physics Auto-Correct (Fix Amalgam & Glare)
                    new_id, physics_mult = self._autocorrect_class(img_np, mask, class_id)
                    
                    if new_id == -1: continue # A fost identificat ca reflexie
                    
                    final_conf = float(score) * physics_mult
                    
                    # 2. Size Check
                    ratio = np.sum(mask) / img_area
                    # Daca e patologie si e uriasa (>15%), e eroare.
                    if new_id < 9 and ratio > 0.15: continue 
                    # Daca e dinte si e minuscul (<0.5%), e zgomot (rezolva numerotarea gresita)
                    if new_id == 9 and ratio < 0.005: continue 

                    if final_conf < conf_threshold: continue

                    raw_detections.append({
                        "id": new_id,
                        "label": self.labels[new_id],
                        "conf": final_conf,
                        "mask": mask
                    })
            except: continue

        return self._hierarchical_processing(raw_detections, width, height)

    def _hierarchical_processing(self, detections, width, height):
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        teeth = [d for d in detections if d['id'] == 9]
        pathologies = [d for d in detections if d['id'] != 9]
        
        # CLEAN UP TEETH (NMS)
        final_teeth = []
        for t in teeth:
            is_dup = False
            for ft in final_teeth:
                inter = np.logical_and(t['mask'], ft['mask']).sum()
                union = np.logical_or(t['mask'], ft['mask']).sum()
                if (inter/union) > 0.30: is_dup = True; break
            if not is_dup: final_teeth.append(t)
            
        # Detect Jaw
        is_upper_jaw = self._detect_jaw_type(final_teeth, height)
        print(f"DEBUG: Jaw -> {'UPPER' if is_upper_jaw else 'LOWER'}")

        # Assign FDI
        final_teeth = self._assign_fdi_notation(final_teeth, width, is_upper_jaw)
        
        # Pathology Matching
        final_paths = []
        for p in pathologies:
            on_tooth = False
            for t in final_teeth:
                overlap = np.logical_and(p['mask'], t['mask']).sum()
                # Relaxam putin conditia de overlap pt carii marginale (0.3 in loc de 0.4)
                if (overlap / p['mask'].sum()) > 0.3: on_tooth = True; break
            
            if on_tooth:
                # NMS pt patologii (Sa nu avem si Carie si Filling in acelasi loc)
                is_dup = False
                for fp in final_paths:
                    inter = np.logical_and(p['mask'], fp['mask']).sum()
                    if (inter / p['mask'].sum()) > 0.5: is_dup = True; break
                if not is_dup: final_paths.append(p)
                
        return final_teeth + final_paths

    def _detect_jaw_type(self, teeth, height):
        if len(teeth) < 4: return True 
        teeth.sort(key=lambda x: np.mean(np.where(x['mask'])[1]))
        
        centroids = [np.mean(np.where(t['mask'])[0]) for t in teeth]
        n = len(centroids)
        
        # Comparam Y-ul Incisivilor (Center) cu Y-ul Molarilor (Outer)
        avg_incisor_y = np.mean(centroids[int(n*0.4):int(n*0.6)+1])
        avg_molar_y = (np.mean(centroids[0:2]) + np.mean(centroids[-2:])) / 2
        
        # Incisivi mai sus (Y mic) decat Molarii -> Maxilar (Arcul e in jos)
        return avg_incisor_y < avg_molar_y

    def _assign_fdi_notation(self, teeth, width, is_upper_jaw):
        # Sortam stanga-dreapta
        teeth.sort(key=lambda x: np.mean(np.where(x['mask'])[1]))
        center_line = width / 2
        
        img_left = [t for t in teeth if np.mean(np.where(t['mask'])[1]) < center_line] 
        img_right = [t for t in teeth if np.mean(np.where(t['mask'])[1]) >= center_line]
        
        if is_upper_jaw:
            img_left.reverse() # 18..11 -> 11..18
            q1, q2 = 1, 2
        else:
            img_left.reverse() # 48..41 -> 41..48
            q1, q2 = 4, 3
            
        for i, t in enumerate(img_left):
            if i >= 8: t['id'] = -1; continue
            t['fdi'] = f"{q1}{i+1}"
            
        for i, t in enumerate(img_right):
            if i >= 8: t['id'] = -1; continue
            t['fdi'] = f"{q2}{i+1}"
            
        return [t for t in teeth if t.get('id') != -1]