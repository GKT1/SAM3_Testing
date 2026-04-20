import json
import matplotlib.pyplot as plt

data_lines = [
    {"epoch": 1, "train_loss": 93.67957836145278, "val_loss": 8.239007609625457},
    {"epoch": 2, "train_loss": 89.21252757020949, "val_loss": 8.014015772063205},
    {"epoch": 3, "train_loss": 87.8068899299303, "val_loss": 7.920158830979453},
    {"epoch": 4, "train_loss": 86.56414370644553, "val_loss": 7.927645298230493},
    {"epoch": 5, "train_loss": 85.91738871110897, "val_loss": 7.91082310193527},
    {"epoch": 6, "train_loss": 85.14255689469758, "val_loss": 7.83986350147838},
    {"epoch": 7, "train_loss": 84.63116235415262, "val_loss": 7.7880788953535465}
]

epochs = [d['epoch'] for d in data_lines]
train_losses = [d['train_loss'] for d in data_lines]
val_losses = [d['val_loss'] for d in data_lines]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Loss', color=color)
ax2.plot(epochs, val_losses, color=color, marker='s', label='Val Loss')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training and Validation Loss Across Epochs')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

output_path = '/teamspace/studios/this_studio/SAM3_Testing/training_loss_plot.png'
plt.savefig(output_path, dpi=200)
print(f"Training loss plot saved to {output_path}")
