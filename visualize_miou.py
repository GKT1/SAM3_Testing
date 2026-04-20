import os
import json
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add SAM3 and SAM3_LoRA to path
project_root = "/teamspace/studios/this_studio/SAM3_Project"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# Paths
JSON_PATH = "/teamspace/studios/this_studio/DisasterM3_Filtered/benchmark_filtered.json"
DATASET_ROOT = "/teamspace/studios/this_studio/DisasterM3/DisasterM3_Bench"
VIS_DIR = "/teamspace/studios/this_studio/SAM3_Testing/visualizations"

# Create output directory
os.makedirs(VIS_DIR, exist_ok=True)

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
    if "building" in desc:
        if "intact" in desc: return "intact building"
        else: return "damaged building"
    if "road" in desc:
        if "intact" in desc: return "intact road"
        else: return "road damage"
    return "damaged building"

def get_ground_truth_mask(image_filename, text_prompt):
    """Loads the ground truth mask based on the text prompt."""
    base_name = os.path.basename(image_filename)
    base_name = base_name.replace("_post_disaster", "").replace("_post", "")
    
    folders = PROMPT_TO_FOLDER.get(text_prompt, [])
    combined_mask = None
    
    for folder in folders:
        mask_path = os.path.join(DATASET_ROOT, "masks", folder, base_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path))
            if len(mask_img.shape) > 2:
                mask_img = mask_img[:, :, 0]
            b_mask = mask_img > 0
            if combined_mask is None:
                combined_mask = b_mask
            else:
                combined_mask = np.logical_or(combined_mask, b_mask)
                
    if combined_mask is None:
        for potential_folder in os.listdir(os.path.join(DATASET_ROOT, "masks")):
            mask_path = os.path.join(DATASET_ROOT, "masks", potential_folder, base_name)
            if os.path.exists(mask_path):
                mask_img = np.array(Image.open(mask_path))
                if len(mask_img.shape) > 2: mask_img = mask_img[:, :, 0]
                return mask_img > 0
        return None
        
    return combined_mask

def build_model(with_lora=False):
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(project_root, "SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True
    )
    
    if with_lora:
        lora_config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            apply_to_vision_encoder=True,
            apply_to_text_encoder=True,
            apply_to_geometry_encoder=False,
            apply_to_detr_encoder=True,
            apply_to_detr_decoder=True,
            apply_to_mask_decoder=False
        )
        model = apply_lora_to_model(model, lora_config)
        weights_path = hf_hub_download(repo_id="GKT/disaster_m3_filtered_base_lora", filename="best_lora_weights.pt")
        lora_state_dict = torch.load(weights_path, map_location="cuda")
        
        cleaned_dict = {}
        for k, v in lora_state_dict.items():
            if isinstance(v, torch.nn.Parameter):
                cleaned_dict[k] = v.data
            else:
                cleaned_dict[k] = v
                
        model.load_state_dict(cleaned_dict, strict=False)
        model = model.to("cuda")
    
    return model

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def run_inference(model, processor, data, device):
    results = {}
    for item in tqdm(data):
        image_rel_path = item['post_image_path'].replace('\\', '/')
        image_path = os.path.join(DATASET_ROOT, image_rel_path)
        if not os.path.exists(image_path): continue
            
        text_prompt = determine_prompt(item['cls_description'])
        gt_mask = get_ground_truth_mask(image_rel_path, text_prompt)
        if gt_mask is None: continue
            
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)
        
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
        results[image_rel_path] = {
            "pred_mask": pred_mask,
            "iou": iou
        }
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # We will visualize 20 images
    NUM_VISUALIZE = 20
    data = data[:NUM_VISUALIZE]
    print(f"Visualizing on {len(data)} samples.")
    
    # Collect necessary image information
    valid_samples = []
    for item in data:
        image_rel_path = item['post_image_path'].replace('\\', '/')
        image_path = os.path.join(DATASET_ROOT, image_rel_path)
        if not os.path.exists(image_path): continue
        text_prompt = determine_prompt(item['cls_description'])
        gt_mask = get_ground_truth_mask(image_rel_path, text_prompt)
        if gt_mask is None: continue
        
        valid_samples.append({
            "image_rel_path": image_rel_path,
            "image_path": image_path,
            "text_prompt": text_prompt,
            "gt_mask": gt_mask,
            "item": item
        })
        
    print(f"Found {len(valid_samples)} valid samples for visualization.")
    
    # 1. Evaluate Base Model
    print("\n--- Running Inference: Base SAM3 Model ---")
    model_base = build_model(with_lora=False)
    processor_base = Sam3Processor(model_base, device=device, confidence_threshold=0.5)
    base_results = run_inference(model_base, processor_base, [s["item"] for s in valid_samples], device)
    
    del model_base
    del processor_base
    torch.cuda.empty_cache()
    
    # 2. Evaluate LoRA Model
    print("\n--- Running Inference: LoRA SAM3 Model ---")
    model_lora = build_model(with_lora=True)
    processor_lora = Sam3Processor(model_lora, device=device, confidence_threshold=0.5)
    lora_results = run_inference(model_lora, processor_lora, [s["item"] for s in valid_samples], device)
    
    del model_lora
    del processor_lora
    torch.cuda.empty_cache()
    
    # 3. Create Plots
    print(f"\n--- Generating Plots in {VIS_DIR} ---")
    for sample in valid_samples:
        rel_path = sample["image_rel_path"]
        if rel_path not in base_results or rel_path not in lora_results:
            continue
            
        base_mask = base_results[rel_path]["pred_mask"]
        base_iou = base_results[rel_path]["iou"]
        
        lora_mask = lora_results[rel_path]["pred_mask"]
        lora_iou = lora_results[rel_path]["iou"]
        
        image = Image.open(sample["image_path"]).convert("RGB")
        gt_mask = sample["gt_mask"]
        prompt = sample["text_prompt"]
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f"Original Image\nPrompt: '{prompt}'")
        axes[0].axis('off')
        
        axes[1].imshow(image)
        axes[1].imshow(gt_mask, alpha=0.5, cmap='Greens')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(image)
        axes[2].imshow(base_mask, alpha=0.5, cmap='Reds')
        axes[2].set_title(f"Base Model Mask\nIoU: {base_iou:.4f}")
        axes[2].axis('off')
        
        axes[3].imshow(image)
        axes[3].imshow(lora_mask, alpha=0.5, cmap='Blues')
        axes[3].set_title(f"LoRA Model Mask\nIoU: {lora_iou:.4f}")
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save
        safe_name = os.path.basename(rel_path).replace('.png', '.jpg')
        save_path = os.path.join(VIS_DIR, f"compare_{safe_name}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Done! Check the '{VIS_DIR}' folder for side-by-side comparisons.")

if __name__ == "__main__":
    main()
