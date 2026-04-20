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
from lora_layers import LoRAConfig, apply_lora_to_model

# Paths
JSON_PATH = "/teamspace/studios/this_studio/DisasterM3_Filtered/benchmark_filtered.json"
DATASET_ROOT = "/teamspace/studios/this_studio/DisasterM3/DisasterM3_Bench"
OUTPUT_JSON = "/teamspace/studios/this_studio/SAM3_Testing/epoch_miou_results.json"
PLOT_PATH = "/teamspace/studios/this_studio/SAM3_Testing/epoch_progression_plot.png"

# Mapping from our 4 text prompts to mask folders
PROMPT_TO_FOLDER = {
    "damaged building": ["test_building_damaged_mask", "test_building_destroyed_mask"],
    "intact building": ["test_building_intact_mask"],
    "road damage": ["test_road_flooded_mask", "test_road_debris_covered_mask"],
    "intact road": ["test_road_intact_mask"]
}

def determine_prompt(cls_description):
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

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_ground_truth_mask(image_filename, text_prompt):
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

def build_model_with_lora():
    """Initializes the base SAM3 model and applies LoRA architecture (weights not loaded yet)."""
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(project_root, "SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True
    )
    
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
    model = model.to("cuda")
    return model

def load_epoch_weights(model, epoch):
    """Downloads and loads weights for a specific epoch from Hugging Face."""
    filename = f"lora_weights_epoch_{epoch}.pt"
    print(f"\n--- Loading {filename} ---")
    weights_path = hf_hub_download(repo_id="GKT/disaster_m3_filtered_base_lora", filename=filename)
    
    lora_state_dict = torch.load(weights_path, map_location="cuda")
    
    cleaned_dict = {}
    for k, v in lora_state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            cleaned_dict[k] = v.data
        else:
            cleaned_dict[k] = v
            
    model.load_state_dict(cleaned_dict, strict=False)

def run_inference(processor, data, device="cuda"):
    """Runs inference on the dataset and returns the mIoU."""
    ious = []
    
    for item in tqdm(data, leave=False):
        image_rel_path = item['post_image_path'].replace('\\', '/')
        image_path = os.path.join(DATASET_ROOT, image_rel_path)
        
        if not os.path.exists(image_path): continue
            
        text_prompt = determine_prompt(item['cls_description'])
        gt_mask = get_ground_truth_mask(image_rel_path, text_prompt)
        
        if gt_mask is None: continue
            
        # Run inference
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
        ious.append(iou)
        
        if len(ious) % 200 == 0:
            print(f"Current Epoch mIoU (after {len(ious)} valid samples): {np.mean(ious):.4f}")
        
    return np.mean(ious) if ious else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Limit to 100 images
    data = data[:100]
    print(f"Evaluating on {len(data)} samples per epoch.")
    
    results = {}
    epochs = list(range(1, 8))
    
    # Instantiate the base model with LoRA layers ONCE
    model = build_model_with_lora()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.5)
    
    # Iterate through all 12 epochs
    for epoch in epochs:
        try:
            load_epoch_weights(model, epoch)
            
            with torch.no_grad():
                miou = run_inference(processor, data, device)
            
            print(f"Epoch {epoch} mIoU: {miou:.4f}")
            results[epoch] = miou
            
        except Exception as e:
            print(f"Failed to evaluate epoch {epoch}: {str(e)}")
            results[epoch] = None
    
    # Save results to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {OUTPUT_JSON}")
    
    # Plot results
    valid_epochs = [e for e, m in results.items() if m is not None]
    valid_mious = [results[e] for e in valid_epochs]
    
    if valid_epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_epochs, valid_mious, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('SAM3 LoRA mIoU Progression Across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(valid_epochs)
        
        # Annotate the points
        for e, m in zip(valid_epochs, valid_mious):
            plt.annotate(f"{m:.4f}", (e, m), textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=200)
        print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()
