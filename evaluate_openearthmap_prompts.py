import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import glob
import re
import argparse
import copy
import time

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
OUTPUT_JSON = "/workspace/SAM3_Testing/openearthmap_prompt_sensitivity_results.json"
WEIGHTS_DIR = "/workspace/SAM3_train_lora/SAM3_LoRA/outputs/open_earth_map_full_lora"

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

PROMPT_VARIANTS = {
    "Bareland": {
        "contextual": "Bareland in satellite imagery",
        "synonym_1": "Dirt",
        "synonym_2": "Barren terrain"
    },
    "Rangeland": {
        "contextual": "Rangeland in aerial view",
        "synonym_1": "Grassland",
        "synonym_2": "Pasture"
    },
    "Developed space": {
        "contextual": "Developed space in satellite imagery",
        "synonym_1": "Urban area",
        "synonym_2": "Built-up area"
    },
    "Road": {
        "contextual": "Road network in aerial view",
        "synonym_1": "Street",
        "synonym_2": "Highway"
    },
    "Tree": {
        "contextual": "Trees in satellite imagery",
        "synonym_1": "Forest",
        "synonym_2": "Woodland"
    },
    "Water": {
        "contextual": "Water body in aerial view",
        "synonym_1": "River",
        "synonym_2": "Lake"
    },
    "Agriculture land": {
        "contextual": "Agriculture land in satellite imagery",
        "synonym_1": "Farmland",
        "synonym_2": "Crop field"
    },
    "Building": {
        "contextual": "Building in remote sensing imagery",
        "synonym_1": "House",
        "synonym_2": "Structure"
    }
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
    """Initializes the base SAM3 model and applies LoRA architecture (weights not loaded yet)."""
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
        
    label_ious = {
        label: {
            # "original": [],
            "contextual": [],
            "synonym_1": [],
            "synonym_2": []
        } for label in COLOR_MAP.keys()
    }
    
    total_predictions = 0
    
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
        base_state = processor.set_image(image)
        
        for label, color in COLOR_MAP.items():
            color_array = np.array(color)
            # Create binary mask for this specific color
            # match shape (H, W, 3) with (3,)
            match = np.all(label_rgb == color_array, axis=-1)
            
            if not np.any(match):
                # This label is not present in this image, skip
                continue
                
            gt_mask = match
            
            prompts_to_test = {
                # "original": label,
                "contextual": PROMPT_VARIANTS[label]["contextual"],
                "synonym_1": PROMPT_VARIANTS[label]["synonym_1"],
                "synonym_2": PROMPT_VARIANTS[label]["synonym_2"]
            }
            
            for p_type, p_text in prompts_to_test.items():
                # Need to use a fresh state for each prompt
                state = copy.deepcopy(base_state)
                state = processor.set_text_prompt(state=state, prompt=p_text)
                total_predictions += 1
                
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
                label_ious[label][p_type].append(iou)
            
    # Calculate means
    label_mious = {}
    for label, prompt_types in label_ious.items():
        label_mious[label] = {}
        for p_type, ious in prompt_types.items():
            label_mious[label][p_type] = np.mean(ious) if ious else 0.0
            
    # Calculate overall mIoU per prompt type across all labels
    overall_miou_by_type = {
        # "original": [],
        "contextual": [],
        "synonym_1": [],
        "synonym_2": []
    }
    
    for label, metrics in label_mious.items():
        for p_type, val in metrics.items():
            overall_miou_by_type[p_type].append(val)
            
    overall_mious = {p_type: np.mean(vals) for p_type, vals in overall_miou_by_type.items()}
    
    return label_mious, overall_mious, total_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenEarthMap on SAM3 with Prompt Sensitivity")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}

    limit_samples = args.limit
    print(f"Limiting to {limit_samples} samples per model.")

    # 2. LoRA Models
    start_epoch = 6
    print("\nEvaluating LoRA Epochs (Prompt Sensitivity)...")
    lora_model = build_model_with_lora()
    processor = Sam3Processor(lora_model, device=device, confidence_threshold=0.4)
    
    # FIX: Helper function to extract integer epoch for proper numerical sorting
    def get_epoch_num(filepath):
        match = re.search(r"lora_weights_epoch_(\d+)\.pt", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    # Fetch and sort numerically
    epoch_weights = sorted(
        glob.glob(os.path.join(WEIGHTS_DIR, "lora_weights_epoch_*.pt")), 
        key=get_epoch_num
    )
    
    for weight_path in epoch_weights:
        epoch = get_epoch_num(weight_path)
        
        # NEW: Skip if the epoch is lower than our designated start_epoch
        if epoch < start_epoch:
            continue
            
        try:
            load_epoch_weights(lora_model, weight_path)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.no_grad():
                label_mious, overall_mious, total_preds = run_inference(processor, device, limit=limit_samples)
            total_time = time.time() - start_time
            avg_inf_time = total_time / total_preds if total_preds > 0 else 0
            
            max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            total_params = sum(p.numel() for p in lora_model.parameters()) / (10 ** 6)
            
            print(f"Epoch {epoch} Prompt Results:")
            # print(f"  Overall mIoU - Original: {overall_mious['original']:.4f}")
            print(f"  Overall mIoU - Contextual: {overall_mious['contextual']:.4f}")
            print(f"  Overall mIoU - Synonym 1: {overall_mious['synonym_1']:.4f}")
            print(f"  Overall mIoU - Synonym 2: {overall_mious['synonym_2']:.4f}")
            
            print(f"\nComplexity Metrics:")
            print(f"  Parameters: {total_params:.2f} M")
            print(f"  Avg Inf. Time: {avg_inf_time:.4f} s/prediction")
            print(f"  Total Time: {total_time:.2f} s (for {total_preds} predictions)")
            print(f"  VRAM Usage: {max_vram:.2f} GB")
            print(f"  GFLOPs: N/A (requires external profiler for dynamic SAM3)")
            
            for label, metrics in label_mious.items():
                print(f"    {label}: Ctx {metrics['contextual']:.4f} | Syn1 {metrics['synonym_1']:.4f} | Syn2 {metrics['synonym_2']:.4f}")

            epoch_result = {
                "overall_by_type": overall_mious,
                "labels": label_mious,
                "complexity": {
                    "Parameters (M)": total_params,
                    "Avg Inf. Time (s)": avg_inf_time,
                    "Total Time (s)": total_time,
                    "Total Predictions": total_preds,
                    "VRAM (GB)": max_vram,
                    "GFLOPs": "N/A"
                }
            }
                
            results[f"epoch_{epoch}"] = epoch_result
            
        except Exception as e:
            print(f"Failed to evaluate epoch {epoch}: {str(e)}")
            results[f"epoch_{epoch}"] = None
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # Update JSON incrementally or merge if file already exists so you don't lose previous data
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try:
                existing_results = json.load(f)
                existing_results.update(results)
                results = existing_results
            except json.JSONDecodeError:
                pass # Overwrite if the file is corrupted
                
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()