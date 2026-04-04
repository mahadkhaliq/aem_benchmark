import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG = os.path.join(os.path.dirname(__file__), 'results', 'nautilus_transformer_run.log')
OUT = os.path.join(os.path.dirname(__file__), 'results', 'transformer_training_curves.png')

epochs, train_losses, val_losses, lr_reduction_epochs = [], [], [], []

with open(LOG) as f:
    for line in f:
        m = re.match(r'This is Epoch (\d+), training loss ([\d.]+), validation loss ([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
        m = re.match(r'Epoch\s+(\d+): reducing learning rate', line)
        if m:
            lr_reduction_epochs.append(int(m.group(1)))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax.plot(epochs, train_losses, label='Train MSE', linewidth=1.8)
ax.plot(epochs, val_losses, label='Val MSE', linewidth=1.8)

for ep in lr_reduction_epochs:
    ax.axvline(x=ep, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Mark best val MSE
best_val = min(val_losses)
best_epoch = epochs[val_losses.index(best_val)]
ax.scatter([best_epoch], [best_val], color='red', zorder=5, label=f'Best val MSE: {best_val:.5f} (epoch {best_epoch})')

# Paper target line
ax.axhline(y=0.001470, color='green', linestyle=':', linewidth=1.5, label='Paper target (0.001470)')

ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Transformer Variant 1 — Training Curves (Nautilus)')
ax.legend()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.grid(True, alpha=0.3)

# Annotate LR reduction lines
for ep in lr_reduction_epochs:
    ax.annotate('LR↓', xy=(ep, min(val_losses) * 1.3), xytext=(ep + 2, min(val_losses) * 1.3),
                fontsize=7, color='gray')

# Zoomed view (epoch 10 onwards)
zoom_start = 1  # skip epoch 0
ax2.plot(epochs[zoom_start:], train_losses[zoom_start:], label='Train MSE', linewidth=1.8)
ax2.plot(epochs[zoom_start:], val_losses[zoom_start:], label='Val MSE', linewidth=1.8)
for ep in lr_reduction_epochs:
    ax2.axvline(x=ep, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.annotate('LR↓', xy=(ep, min(val_losses) * 1.3), xytext=(ep + 2, min(val_losses) * 1.3),
                 fontsize=7, color='gray')
ax2.scatter([best_epoch], [best_val], color='red', zorder=5, label=f'Best val MSE: {best_val:.5f} (epoch {best_epoch})')
ax2.axhline(y=0.001470, color='green', linestyle=':', linewidth=1.5, label='Paper target (0.001470)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Zoomed — Epoch 10 onwards')
ax2.legend(fontsize=8)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.show()
print(f"Saved to {OUT}")
