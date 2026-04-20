import os
import torch
import json
import sys

# Add SAM3 to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SAM3_Project/sam3")))
from sam3.model_builder import build_sam3_image_model
from mqa_evaluator import evaluate_mqa_on_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SAM3_Project")))
from SAM3_LoRA.lora_layers import apply_lora_to_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MQA Scenarios
    bdc_scenarios_path = "../SAM3_Project/test_data/100_images_test/common_samples/sample_BDC.json"
    rdc_scenarios_path = "../SAM3_Project/test_data/100_images_test/common_samples/sample_RDC.json"
    images_dir = "../SAM3_Project/test_data/100_images_test/test_images"

    # # 1. Base Model
    # print("\n--- Loading Base SAM3 ---")
    # model = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True)
    # model.to(device)
    # model.eval()

    # print("\nEvaluating Base Model on BDC...")
    # bdc_base = evaluate_mqa_on_dataset(model, device.type, bdc_scenarios_path, images_dir)
    # print(f"Base BDC -> Acc: {bdc_base['accuracy']:.2%}, MAE: {bdc_base['mae']:.2f}")

    # print("\nEvaluating Base Model on RDC...")
    # rdc_base = evaluate_mqa_on_dataset(model, device.type, rdc_scenarios_path, images_dir)
    # print(f"Base RDC -> Acc: {rdc_base['accuracy']:.2%}, MAE: {rdc_base['mae']:.2f}")

    # # Free up memory
    # del model
    # torch.cuda.empty_cache()

    # 2. LoRA Model
    print("\n--- Loading SAM3 + LoRA ---")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SAM3_Project/SAM3_LoRA")))
    from lora_layers import LoRAConfig, load_lora_weights
    from huggingface_hub import hf_hub_download
    
    model = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True)
    
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
    
    weights_path = hf_hub_download(repo_id="GKT/disaster_m3_filtered_base_lora", filename="lora_weights_epoch_3.pt")
    lora_state_dict = torch.load(weights_path, map_location="cpu")
    
    cleaned_dict = {}
    for k, v in lora_state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            cleaned_dict[k] = v.data
        else:
            cleaned_dict[k] = v
            
    model.load_state_dict(cleaned_dict, strict=False)
    model = model.to(device)
    
    model.eval()

    print("\nEvaluating LoRA Model on BDC...")
    bdc_lora = evaluate_mqa_on_dataset(model, device.type, bdc_scenarios_path, images_dir)
    print(f"LoRA BDC -> Acc: {bdc_lora['accuracy']:.2%}, MAE: {bdc_lora['mae']:.2f}")

    print("\nEvaluating LoRA Model on RDC...")
    rdc_lora = evaluate_mqa_on_dataset(model, device.type, rdc_scenarios_path, images_dir)
    print(f"LoRA RDC -> Acc: {rdc_lora['accuracy']:.2%}, MAE: {rdc_lora['mae']:.2f}")

    print("\n--- Summary ---")
    print("BDC (Building Damage Counting):")
    # print(f"  Base -> Acc: {bdc_base['accuracy']:.2%}, MAE: {bdc_base['mae']:.2f}")
    print(f"  LoRA -> Acc: {bdc_lora['accuracy']:.2%}, MAE: {bdc_lora['mae']:.2f}")
    print("RDC (Road Damage Counting):")
    # print(f"  Base -> Acc: {rdc_base['accuracy']:.2%}, MAE: {rdc_base['mae']:.2f}")
    print(f"  LoRA -> Acc: {rdc_lora['accuracy']:.2%}, MAE: {rdc_lora['mae']:.2f}")


if __name__ == "__main__":
    main()
