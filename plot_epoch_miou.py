import json
import matplotlib.pyplot as plt

INPUT_JSON = "/teamspace/studios/this_studio/SAM3_Testing/epoch_miou_results.json"
PLOT_PATH = "/teamspace/studios/this_studio/SAM3_Testing/epoch_miou_plot.png"

def main():
    try:
        with open(INPUT_JSON, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading {INPUT_JSON}: {e}")
        return

    valid_epochs = []
    valid_mious = []
    
    # Sort keys as integers
    sorted_epochs = sorted([int(k) for k in results.keys()])
    
    for e in sorted_epochs:
        m = results[str(e)]
        if m is not None:
            valid_epochs.append(e)
            valid_mious.append(m)

    if valid_epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_epochs, valid_mious, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('SAM3 LoRA mIoU Progression Across 7 Epochs (100 Samples)')
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
    else:
        print("No valid evaluation data found.")

if __name__ == "__main__":
    main()