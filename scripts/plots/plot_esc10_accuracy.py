from pathlib import Path

import matplotlib.pyplot as plt

epochs_esc10 = list(range(1, 21))
train_acc_esc10 = [23.44,35.31,40.00,46.56,50.94,55.94,67.50,72.81,79.69,81.25,79.06,83.75,85.00,84.06,87.81,91.88,91.56,90.62,90.31,91.25]
val_acc_esc10   = [42.50,25.00,42.50,66.25,56.25,60.00,67.50,70.00,71.25,75.00,72.50,71.25,75.00,77.50,77.50,76.25,78.75,76.25,76.25,77.50]
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(epochs_esc10, train_acc_esc10, marker="o", label="Training accuracy")
plt.plot(epochs_esc10, val_acc_esc10, marker="s", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ESC-10 accuracy")
plt.xticks(epochs_esc10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "esc10_accuracy_curve.png", dpi=300)
plt.show()
