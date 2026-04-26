import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import glob

# Add SAM3 and SAM3_LoRA to path
project_root = "/workspace"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "SAM3_train_lora", "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from lora_layers import LoRAConfig, apply_lora_to_model

# Paths
IMAGE_DIR = "/workspace/Small_OpenEarthMap_Test/images"
LABEL_DIR = "/workspace/Small_OpenEarthMap_Test/labels"
OUTPUT_JSON = "/workspace/SAM3_Testing/openearthmap_threshold_results_epoch9.json"
WEIGHT_PATH = "/workspace/SAM3_train_lora/SAM3_LoRA/outputs/open_earth_map_full_lora/lora_weights_epoch_9.pt"

COLOR_MAP = {
    "Bareland": [128, 0, 0],
    "Rangeland": [0, 255, 36],
    "Developed space": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Agriculture land": [75, 181, 73],
    "Building": [222, 31, 7]
}

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def build_base_model():
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(project_root, "SAM3_train_lora/SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True
    )
    return model

def build_model_with_lora():
    """Initializes the base SAM3 model and applies LoRA architecture."""
    model = build_base_model()
    
    # Matching light config
    lora_config = LoRAConfig(
        rank=32,
        alpha=32,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj", "fc1","fc2","c_fc","c_proj","linear1","linear2" ],
        apply_to_vision_encoder=True,
        apply_to_text_encoder=True,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=True
    )
    model = apply_lora_to_model(model, lora_config)
    model = model.to("cuda")
    return model

def load_epoch_weights(model, weights_path):
    print(f"\n--- Loading {os.path.basename(weights_path)} ---")
    lora_state_dict = torch.load(weights_path, map_location="cuda")
    
    cleaned_dict = {}
    for k, v in lora_state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            cleaned_dict[k] = v.data
        else:
            cleaned_dict[k] = v
            
    model.load_state_dict(cleaned_dict, strict=False)

def run_inference(processor, device="cuda", limit=None):
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg') or f.endswith('.png')])
    
    if limit is not None:
        image_files = image_files[:limit]
        
    label_ious = {label: [] for label in COLOR_MAP.keys()}
    
    for img_name in tqdm(image_files, leave=False):
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_name = img_name.rsplit('.', 1)[0] + '.png'
        label_path = os.path.join(LABEL_DIR, label_name)
        
        if not os.path.exists(label_path):
            continue
            
        # Load image and ground truth label
        image = Image.open(img_path).convert("RGB")
        label_img = np.array(Image.open(label_path)) # RGBA
        if label_img.shape[-1] == 4:
            label_rgb = label_img[:, :, :3]
        else:
            label_rgb = label_img
            
        # Set image in processor once per image
        state = processor.set_image(image)
        
        for label, color in COLOR_MAP.items():
            color_array = np.array(color)
            # Create binary mask for this specific color
            match = np.all(label_rgb == color_array, axis=-1)
            
            if not np.any(match):
                # This label is not present in this image, skip
                continue
                
            gt_mask = match
            
            # Predict
            state = processor.set_text_prompt(state=state, prompt=label)
            
            if state["masks"].shape[0] > 0:
                pred_mask_tensor = state["masks"].squeeze(1).sum(dim=0) > 0
                pred_mask = pred_mask_tensor.cpu().numpy()
                
                if pred_mask.shape != gt_mask.shape:
                    pred_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                    pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    pred_mask = np.array(pred_img) > 128
            else:
                pred_mask = np.zeros_like(gt_mask, dtype=bool)
                
            iou = calculate_iou(pred_mask, gt_mask)
            label_ious[label].append(iou)
            
    # Calculate means
    label_mious = {label: (np.mean(ious) if ious else 0.0) for label, ious in label_ious.items()}
    overall_miou = np.mean([m for m in label_mious.values()])
    
    return label_mious, overall_miou

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}

    limit_samples = None
    print(f"Limiting to {limit_samples} samples per model.")
    
    print("\nLoading LoRA Model...")
    lora_model = build_model_with_lora()
    
    try:
        load_epoch_weights(lora_model, WEIGHT_PATH)
    except Exception as e:
        print(f"Failed to load weights from {WEIGHT_PATH}: {str(e)}")
        return
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\nEvaluating with confidence_threshold = {threshold}")
        # Initialize processor with specific threshold
        processor = Sam3Processor(lora_model, device=device, confidence_threshold=threshold)
        
        try:
            with torch.no_grad():
                label_mious, overall_miou = run_inference(processor, device, limit=limit_samples)
            
            print(f"Threshold {threshold} Overall mIoU: {overall_miou:.4f}")
            for label, miou in label_mious.items():
                print(f"  {label}: {miou:.4f}")
                
            results[f"threshold_{threshold}"] = {
                "overall": overall_miou,
                "labels": label_mious
            }
            
        except Exception as e:
            print(f"Failed to evaluate threshold {threshold}: {str(e)}")
            results[f"threshold_{threshold}"] = None
            
        # Optional: Save results progressively inside the loop so we don't lose data if it crashes
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
            
    print(f"\nResults saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
