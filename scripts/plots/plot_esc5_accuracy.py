from pathlib import Path

import matplotlib.pyplot as plt

epochs_esc5 = list(range(1, 17))
train_acc_esc5 = [32.50,46.25,39.38,50.62,54.38,55.62,69.38,75.62,78.75,82.50,85.00,88.12,86.88,95.00,86.25,93.75]
val_acc_esc5   = [47.50,52.50,60.00,65.00,57.50,80.00,77.50,85.00,87.50,80.00,90.00,87.50,85.00,87.50,85.00,87.50]
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(epochs_esc5, train_acc_esc5, marker="o", label="Training accuracy")
plt.plot(epochs_esc5, val_acc_esc5, marker="s", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ESC-5 accuracy")
plt.xticks(epochs_esc5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "esc5_accuracy_curve.png", dpi=300)
plt.show()
