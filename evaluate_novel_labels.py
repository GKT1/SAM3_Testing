import os
import sys
import torch
import numpy as np
from PIL import Image
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
DATASET_ROOT = "/teamspace/studios/this_studio/DisasterM3/DisasterM3_Bench"

NOVEL_TASKS = {
    "lava": "volcano_lava",
    "flood": "flooding_mask"
}

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_image_path(mask_filename):
    """Converts a mask filename like 'guatemala_volcano_00000001.png' to test image path."""
    base_name = os.path.splitext(mask_filename)[0]
    # In DisasterM3 test_images, they typically have _post_disaster suffix
    test_img_name = f"{base_name}_post_disaster.png"
    return os.path.join(DATASET_ROOT, "test_images", test_img_name)

def load_mask(mask_path):
    mask_img = np.array(Image.open(mask_path))
    if len(mask_img.shape) > 2:
        mask_img = mask_img[:, :, 0] # Take first channel if RGB
    return mask_img > 0

def build_model(with_lora=False):
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(project_root, "SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True
    )
    
    if with_lora:
        # Use exact config from disaster_m3_filtered_light_lora.yaml
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
        
        # Download from HuggingFace
        print("Downloading LoRA weights from HuggingFace...")
        weights_path = hf_hub_download(repo_id="GKT/disaster_m3_filtered_base_lora", filename="lora_weights_epoch_3.pt")
        print(f"Loading LoRA weights from {weights_path}...")
        lora_state_dict = torch.load(weights_path, map_location="cuda")
        
        # Ensure we only pass data tensors to load_state_dict
        cleaned_dict = {}
        for k, v in lora_state_dict.items():
            if isinstance(v, torch.nn.Parameter):
                cleaned_dict[k] = v.data
            else:
                cleaned_dict[k] = v
                
        model.load_state_dict(cleaned_dict, strict=False)
        model = model.to("cuda")
    
    return model

def evaluate_model(model_type, processor, tasks):
    results = {}
    
    print(f"\n--- Evaluating {model_type} Model on Novel Labels ---")
    
    for prompt, folder in tasks.items():
        print(f"\nEvaluating prompt: '{prompt}' (folder: {folder})")
        mask_dir = os.path.join(DATASET_ROOT, "masks", folder)
        
        if not os.path.exists(mask_dir):
            print(f"Folder not found: {mask_dir}")
            continue
            
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')]
        print(f"Found {len(mask_files)} masks for '{prompt}'")
        
        ious = []
        for mask_file in tqdm(mask_files):
            mask_path = os.path.join(mask_dir, mask_file)
            image_path = get_image_path(mask_file)
            
            # Some datasets might use slightly different naming conventions.
            # If the post_disaster one doesn't exist, we try the exact same name as the mask.
            if not os.path.exists(image_path):
                alt_image_path = os.path.join(DATASET_ROOT, "test_images", mask_file)
                if os.path.exists(alt_image_path):
                    image_path = alt_image_path
                else:
                    continue # Image missing
                    
            gt_mask = load_mask(mask_path)
            
            # Run inference
            image = Image.open(image_path).convert("RGB")
            state = processor.set_image(image)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            
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
            
        mean_iou = np.mean(ious) if ious else 0.0
        results[prompt] = {
            "miou": mean_iou,
            "count": len(ious)
        }
        print(f"{model_type} mIoU for '{prompt}': {mean_iou:.4f} (over {len(ious)} valid samples)")
        
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Evaluate Base
    model_base = build_model(with_lora=False)
    processor_base = Sam3Processor(model_base, device=device, confidence_threshold=0.5)
    base_results = evaluate_model("Base SAM3", processor_base, NOVEL_TASKS)
    
    del processor_base
    del model_base
    torch.cuda.empty_cache()
    
    # Evaluate LoRA
    model_lora = build_model(with_lora=True)
    processor_lora = Sam3Processor(model_lora, device=device, confidence_threshold=0.5)
    lora_results = evaluate_model("LoRA SAM3", processor_lora, NOVEL_TASKS)
    
    # Summary
    print("\n================ FINAL NOVEL LABELS SUMMARY ================")
    for prompt in NOVEL_TASKS.keys():
        b_miou = base_results.get(prompt, {}).get("miou", 0)
        l_miou = lora_results.get(prompt, {}).get("miou", 0)
        count = base_results.get(prompt, {}).get("count", 0)
        print(f"[{prompt.upper()}] (N={count}):")
        print(f"  Base Model: {b_miou:.4f}")
        print(f"  LoRA Model: {l_miou:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
