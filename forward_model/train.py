import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import config
from AEML.data import ADM
from AEML.models.MLP import DukeMLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'models', 'MLP', 'adm_mlp')


def evaluate(net, loader, device):
    net.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = net(x)
            total_loss += nn.functional.mse_loss(pred, y, reduction='sum').item()
            n += y.numel()
    return total_loss / n


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.chdir(config.DATA_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_x, test_y = ADM(
        normalize=config.NORMALIZE_INPUT,
        batch_size=config.BATCH_SIZE,
    )

    model = DukeMLP(
        dim_g=14,
        dim_s=2001,
        linear=config.LINEAR,
        skip_connection=False,
        skip_head=0,
        dropout=0,
        model_name='adm_mlp',
    )
    net = model.model.to(device)

    optimizer = Adam(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    # Matches AEML loop: step on val_mse every epoch, patience=10 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_DECAY_RATE,
                                  patience=10, threshold=1e-4)
    writer = SummaryWriter(CKPT_DIR)

    best_val_loss = float('inf')
    start = time.time()

    for epoch in range(config.EPOCHS):
        # Training
        net.train()
        train_loss, n_train = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            loss = nn.functional.mse_loss(pred, y, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += y.numel()

        train_mse = train_loss / n_train

        # Validate every epoch — matches AEML loop exactly
        val_mse = evaluate(net, val_loader, device)
        scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_mse, epoch)
        writer.add_scalar('Loss/val', val_mse, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if epoch % config.EVAL_STEP == 0:
            elapsed = (time.time() - start) / 60
            print(f"Epoch {epoch:4d} | train {train_mse:.5f} | val {val_mse:.5f} | lr {current_lr:.2e} | {elapsed:.1f} min")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(net, os.path.join(CKPT_DIR, 'best_model_forward.pt'))
            if epoch % config.EVAL_STEP == 0:
                print(f"           -> saved (best val {best_val_loss:.5f})")

        if best_val_loss < config.STOP_THRESHOLD:
            print(f"Stopping early at epoch {epoch}")
            break

    # Final test evaluation
    net.eval()
    test_x_t = torch.tensor(test_x).to(device)
    test_y_t = torch.tensor(test_y).to(device)
    with torch.no_grad():
        pred = net(test_x_t)
        test_mse = nn.functional.mse_loss(pred, test_y_t).item()

    writer.add_scalar('Loss/test_final', test_mse, config.EPOCHS)
    writer.close()

    results = {
        'test_mse': test_mse,
        'best_val_mse': best_val_loss,
        'epochs': config.EPOCHS,
        'lr': config.LR,
        'weight_decay': config.WEIGHT_DECAY,
        'linear': config.LINEAR,
    }
    with open(os.path.join(CKPT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTest MSE:      {test_mse:.6f}")
    print(f"Best val MSE:  {best_val_loss:.6f}")
    print(f"Results saved to {CKPT_DIR}/results.json")


if __name__ == '__main__':
    main()
