import os
import json
import sys
import torch
import numpy as np
import yaml
import glob
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add SAM3 and SAM3_LoRA to path
project_root = "/workspace/SAM3_train_lora"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# Paths
JSON_PATH = "/workspace/DisasterM3_road_building_Optical/benchmark_filtered.json"
DATASET_ROOT = "/workspace/DisasterM3/DisasterM3_Bench"

# Mapping from our 4 text prompts to mask folders
PROMPT_TO_FOLDER = {
    "damaged building": ["test_building_damaged_mask", "test_building_destroyed_mask"],
    "intact building": ["test_building_intact_mask"],
    "road damage": ["test_road_flooded_mask", "test_road_debris_covered_mask"],
    "intact road": ["test_road_intact_mask"]
}

def determine_prompt(cls_description):
    """Determine one of the 4 text prompts from the JSON cls_description."""
    desc = cls_description.lower()
    if "intact building" in desc:
        return "intact building"
    elif "damaged building" in desc or "destroyed building" in desc:
        return "damaged building"
    elif "intact road" in desc:
        return "intact road"
    elif "road damage" in desc or "flooded road" in desc or "debris covered road" in desc or "impassable road" in desc:
        return "road damage"
    # fallback heuristics
    if "building" in desc:
        if "intact" in desc: return "intact building"
        else: return "damaged building"
    if "road" in desc:
        if "intact" in desc: return "intact road"
        else: return "road damage"
    
    # default fallback
    return "damaged building"

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_ground_truth_mask(image_filename, text_prompt):
    """Loads the ground truth mask based on the text prompt."""
    # e.g., 'test_images/bata_explosion_post_0.png' -> 'bata_explosion_0.png'
    base_name = os.path.basename(image_filename)
    base_name = base_name.replace("_post_disaster", "").replace("_post", "")
    
    folders = PROMPT_TO_FOLDER.get(text_prompt, [])
    
    # Initialize a combined mask to False
    combined_mask = None
    
    for folder in folders:
        mask_path = os.path.join(DATASET_ROOT, "masks", folder, base_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path))
            if len(mask_img.shape) > 2:
                mask_img = mask_img[:, :, 0] # Take first channel if RGB
            b_mask = mask_img > 0
            if combined_mask is None:
                combined_mask = b_mask
            else:
                combined_mask = np.logical_or(combined_mask, b_mask)
                
    if combined_mask is None:
        # fallback, check if any mask exists with that name anywhere
        for potential_folder in os.listdir(os.path.join(DATASET_ROOT, "masks")):
            mask_path = os.path.join(DATASET_ROOT, "masks", potential_folder, base_name)
            if os.path.exists(mask_path):
                mask_img = np.array(Image.open(mask_path))
                if len(mask_img.shape) > 2: mask_img = mask_img[:, :, 0]
                return mask_img > 0
        return None # No mask found at all
        
    return combined_mask

def get_lora_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    lora_cfg = config_dict.get('lora', {})
    
    return LoRAConfig(
        rank=lora_cfg.get('rank', 8),
        alpha=lora_cfg.get('alpha', 16),
        dropout=lora_cfg.get('dropout', 0.0),
        target_modules=lora_cfg.get('target_modules', ["q_proj", "k_proj", "v_proj", "out_proj"]),
        apply_to_vision_encoder=lora_cfg.get('apply_to_vision_encoder', True),
        apply_to_text_encoder=lora_cfg.get('apply_to_text_encoder', True),
        apply_to_geometry_encoder=lora_cfg.get('apply_to_geometry_encoder', False),
        apply_to_detr_encoder=lora_cfg.get('apply_to_detr_encoder', True),
        apply_to_detr_decoder=lora_cfg.get('apply_to_detr_decoder', True),
        apply_to_mask_decoder=lora_cfg.get('apply_to_mask_decoder', False)
    )

def build_model(with_lora=False, lora_weights_path=None, lora_config=None):
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(project_root, "SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True
    )
    
    if with_lora and lora_weights_path and lora_config:
        model = apply_lora_to_model(model, lora_config)
        
        print(f"Loading LoRA weights from {lora_weights_path}...")
        lora_state_dict = torch.load(lora_weights_path, map_location="cuda")
        
        # Ensure we only pass data tensors to load_state_dict to avoid Parameter reassignment bugs
        cleaned_dict = {}
        for k, v in lora_state_dict.items():
            if isinstance(v, torch.nn.Parameter):
                cleaned_dict[k] = v.data
            else:
                cleaned_dict[k] = v
                
        model.load_state_dict(cleaned_dict, strict=False)
        model = model.to("cuda")
    
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Use full dataset
    print(f"Evaluating on all {len(data)} samples.")
    
    # 1. Evaluate Base Model
    print("\n--- Evaluating Base SAM3 Model ---")
    model_base = build_model(with_lora=False)
    processor = Sam3Processor(model_base, device=device, confidence_threshold=0.5)
    
    ious_base = []
    base_prompt_ious = {p: [] for p in PROMPT_TO_FOLDER.keys()}
    
    for i, item in enumerate(tqdm(data)):
        # Normalize path for cross-platform matching
        image_rel_path = item['post_image_path'].replace('\\', '/')
        image_path = os.path.join(DATASET_ROOT, image_rel_path)
        
        if not os.path.exists(image_path):
            continue
            
        text_prompt = determine_prompt(item['cls_description'])
        gt_mask = get_ground_truth_mask(image_rel_path, text_prompt)
        
        if gt_mask is None:
            continue
            
        # Run inference
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)
        
        if state["masks"].shape[0] > 0:
            # Combine all instance masks into one binary mask
            pred_mask_tensor = state["masks"].squeeze(1).sum(dim=0) > 0
            pred_mask = pred_mask_tensor.cpu().numpy()
            
            # The prediction might need resizing
            if pred_mask.shape != gt_mask.shape:
                pred_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                pred_mask = np.array(pred_img) > 128
        else:
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
                
        iou = calculate_iou(pred_mask, gt_mask)
        ious_base.append(iou)
        base_prompt_ious[text_prompt].append(iou)
        
        if len(ious_base) % 200 == 0:
            print(f"\n--- Current Base mIoU (after {len(ious_base)} valid samples) ---")
            print(f"Overall mIoU: {np.mean(ious_base):.4f}")
            for p, arr in base_prompt_ious.items():
                if arr: print(f"  {p}: {np.mean(arr):.4f} ({len(arr)} samples)")
            
    miou_base = np.mean(ious_base) if ious_base else 0.0
    print(f"\n--- Final Base Model mIoU ---")
    print(f"Overall: {miou_base:.4f} (over {len(ious_base)} samples)")
    for p, arr in base_prompt_ious.items():
        if arr: print(f"  {p}: {np.mean(arr):.4f} ({len(arr)} samples)")
    
    # Free memory
    del model_base
    torch.cuda.empty_cache()
    
    # 2. Evaluate LoRA Model
    print("\n--- Evaluating LoRA SAM3 Model across Epochs ---")
    
    LORA_YAML_CONFIG = "/workspace/SAM3_train_lora/SAM3_LoRA/configs/open_earth_map_full_lora.yaml"
    LORA_OUTPUTS_DIR = "/workspace/SAM3_train_lora/SAM3_LoRA/outputs/open_earth_map_full_lora_final"
    
    print(f"Using config: {LORA_YAML_CONFIG}")
    print(f"Looking for checkpoints in: {LORA_OUTPUTS_DIR}")
    
    lora_config = get_lora_config_from_yaml(LORA_YAML_CONFIG)
    
    epoch_checkpoints = glob.glob(os.path.join(LORA_OUTPUTS_DIR, "lora_weights_epoch_*.pt"))
    
    def extract_epoch(filepath):
        # Extract the epoch number from filename like "lora_weights_epoch_3.pt"
        base = os.path.basename(filepath)
        try:
            return int(base.split("epoch_")[1].split(".pt")[0])
        except:
            return 999
            
    epoch_checkpoints = sorted(epoch_checkpoints, key=extract_epoch)
    
    if not epoch_checkpoints:
        print(f"No checkpoint files found in {LORA_OUTPUTS_DIR}")
        return
        
    print(f"Found {len(epoch_checkpoints)} epoch checkpoints to evaluate.")
    
    epoch_mious = {}
    
    for ckpt_path in epoch_checkpoints:
        epoch_num = extract_epoch(ckpt_path)
        print(f"\n--- Evaluating Epoch {epoch_num} ---")
        
        model_lora = build_model(with_lora=True, lora_weights_path=ckpt_path, lora_config=lora_config)
        processor_lora = Sam3Processor(model_lora, device=device, confidence_threshold=0.5)
        
        ious_lora = []
        lora_prompt_ious = {p: [] for p in PROMPT_TO_FOLDER.keys()}
        
        for i, item in enumerate(tqdm(data)):
            image_rel_path = item['post_image_path'].replace('\\', '/')
            image_path = os.path.join(DATASET_ROOT, image_rel_path)
            
            if not os.path.exists(image_path): continue
                
            text_prompt = determine_prompt(item['cls_description'])
            gt_mask = get_ground_truth_mask(image_rel_path, text_prompt)
            if gt_mask is None: continue
                
            # Run inference sequentially
            image = Image.open(image_path).convert("RGB")
            state = processor_lora.set_image(image)
            state = processor_lora.set_text_prompt(state=state, prompt=text_prompt)
            
            if state["masks"].shape[0] > 0:
                # Combine all instance masks into one binary mask
                pred_mask_tensor = state["masks"].squeeze(1).sum(dim=0) > 0
                pred_mask = pred_mask_tensor.cpu().numpy()
                
                if pred_mask.shape != gt_mask.shape:
                    pred_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                    pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    pred_mask = np.array(pred_img) > 128
            else:
                pred_mask = np.zeros_like(gt_mask, dtype=bool)
                    
            iou = calculate_iou(pred_mask, gt_mask)
            ious_lora.append(iou)
            lora_prompt_ious[text_prompt].append(iou)
            
            if len(ious_lora) % 200 == 0:
                print(f"\n--- Epoch {epoch_num} Current LoRA mIoU (after {len(ious_lora)} valid samples) ---")
                print(f"Overall mIoU: {np.mean(ious_lora):.4f}")
                for p, arr in lora_prompt_ious.items():
                    if arr: print(f"  {p}: {np.mean(arr):.4f} ({len(arr)} samples)")
            
        miou_lora = np.mean(ious_lora) if ious_lora else 0.0
        print(f"\n--- Final Epoch {epoch_num} LoRA Model mIoU ---")
        print(f"Overall: {miou_lora:.4f} (over {len(ious_lora)} samples)")
        for p, arr in lora_prompt_ious.items():
            if arr: print(f"  {p}: {np.mean(arr):.4f} ({len(arr)} samples)")
        
        epoch_mious[f"epoch_{epoch_num}"] = miou_lora
        
        # Free memory before next epoch
        del model_lora
        del processor_lora
        torch.cuda.empty_cache()
    
    print("\n--- Multi-Epoch Summary ---")
    for epoch, miou in epoch_mious.items():
        print(f"{epoch}: mIoU = {miou:.4f}")
        
    # Save results to a file
    results_file = os.path.join("SAM3_Testing", "epoch_mious_results_custom.json")
    with open(results_file, 'w') as f:
        json.dump(epoch_mious, f, indent=4)
    print(f"\nSaved epoch evaluation results to {results_file}")

if __name__ == "__main__":
    main()
