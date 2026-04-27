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
OUTPUT_JSON = "/workspace/SAM3_Testing/openearthmap_epoch_results_full.json"
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
            # match shape (H, W, 3) with (3,)
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
    parser = argparse.ArgumentParser(description="Evaluate OpenEarthMap on SAM3")
    parser.add_argument("--unseen_classes", nargs="+", default=[], help="List of unseen classes to calculate u-mIoU and h-mIoU")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}

    limit_samples = args.limit
    print(f"Limiting to {limit_samples} samples per model.")
    if args.unseen_classes:
        unseen_classes_lower = [c.lower() for c in args.unseen_classes]
        seen_classes_display = [k for k in COLOR_MAP.keys() if k.lower() not in unseen_classes_lower]
        unseen_classes_display = [k for k in COLOR_MAP.keys() if k.lower() in unseen_classes_lower]
        print(f"Configured Seen classes: {seen_classes_display}")
        print(f"Configured Unseen classes: {unseen_classes_display}")

    # 1. Base Model
    # print("\nEvaluating Base Model...")
    # base_model = build_base_model().to(device)
    # processor = Sam3Processor(base_model, device=device, confidence_threshold=0.5)
    
    
    
    # with torch.no_grad():
    #     label_mious, overall_miou = run_inference(processor, device, limit=limit_samples)
        
    # print(f"Base Model Overall mIoU: {overall_miou:.4f}")
    # for label, miou in label_mious.items():
    #     print(f"  {label}: {miou:.4f}")
        
    # results["base"] = {
    #     "overall": overall_miou,
    #     "labels": label_mious
    # }
    
    # # Free base model
    # del base_model
    # del processor
    # torch.cuda.empty_cache()
    
    # 2. LoRA Models
    start_epoch = 6
    print("\nEvaluating LoRA Epochs...")
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
            
            with torch.no_grad():
                label_mious, overall_miou = run_inference(processor, device, limit=limit_samples)
            
            print(f"Epoch {epoch} Overall mIoU: {overall_miou:.4f}")
            for label, miou in label_mious.items():
                print(f"  {label}: {miou:.4f}")
                
            # Compute s-mIoU, u-mIoU, h-mIoU
            s_miou, u_miou, h_miou = None, None, None
            
            if args.unseen_classes:
                unseen_classes_lower = [c.lower() for c in args.unseen_classes]
                seen_classes = [k for k in label_mious.keys() if k.lower() not in unseen_classes_lower]
                unseen_classes = [k for k in label_mious.keys() if k.lower() in unseen_classes_lower]
                
                s_miou = np.mean([label_mious[k] for k in seen_classes]) if seen_classes else 0.0
                u_miou = np.mean([label_mious[k] for k in unseen_classes]) if unseen_classes else 0.0
                
                if s_miou + u_miou > 0:
                    h_miou = (2 * s_miou * u_miou) / (s_miou + u_miou)
                else:
                    h_miou = 0.0
                    
                print(f"  Seen classes: {seen_classes}")
                print(f"  Unseen classes: {unseen_classes}")
                print(f"  Seen mIoU (s-mIoU): {s_miou:.4f}")
                print(f"  Unseen mIoU (u-mIoU): {u_miou:.4f}")
                print(f"  Harmonic mIoU (h-mIoU): {h_miou:.4f}")

            epoch_result = {
                "overall": overall_miou,
                "labels": label_mious
            }
            if args.unseen_classes:
                epoch_result["s_miou"] = s_miou
                epoch_result["u_miou"] = u_miou
                epoch_result["h_miou"] = h_miou
                
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
