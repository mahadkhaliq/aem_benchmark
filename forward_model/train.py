import os
import json
import torch
import torch.nn as nn

import config
from AEML.data import ADM
from AEML.models.MLP.class_wrapper import Network
from AEML.models.MLP.model_maker import Forward

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'models', 'MLP', 'adm_mlp')


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.chdir(config.DATA_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_x, test_y = ADM(
        normalize=config.NORMALIZE_INPUT,
        batch_size=config.BATCH_SIZE,
    )

    # Use the original AEML Network class and training loop exactly as in the paper
    ntwk = Network(
        dim_g=14,
        dim_s=2001,
        linear=config.LINEAR,
        skip_connection=False,
        skip_head=0,
        dropout=0,
        model_name='adm_mlp',
        ckpt_dir=os.path.join(os.path.dirname(__file__), 'models', 'MLP'),
    )

    ntwk.train_(
        train_loader=train_loader,
        test_loader=val_loader,
        epochs=config.EPOCHS,
        optm='Adam',
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_name='reduce_plateau',
        lr_decay_rate=config.LR_DECAY_RATE,
        eval_step=config.EVAL_STEP,
        stop_threshold=config.STOP_THRESHOLD,
    )

    # Final test evaluation — load best saved checkpoint, not last epoch
    net = torch.load(os.path.join(CKPT_DIR, 'best_model_forward.pt'), map_location=device)
    net.eval()
    test_x_t = torch.tensor(test_x).to(device)
    test_y_t = torch.tensor(test_y).to(device)
    with torch.no_grad():
        pred = net(test_x_t)
        test_mse = nn.functional.mse_loss(pred, test_y_t).item()

    results = {
        'test_mse': test_mse,
        'best_val_mse': ntwk.best_validation_loss,
        'epochs': config.EPOCHS,
        'lr': config.LR,
        'weight_decay': config.WEIGHT_DECAY,
        'linear': config.LINEAR,
    }
    with open(os.path.join(CKPT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTest MSE:      {test_mse:.6f}")
    print(f"Best val MSE:  {ntwk.best_validation_loss:.6f}")
    print(f"Results saved to {CKPT_DIR}/results.json")


if __name__ == '__main__':
    main()
