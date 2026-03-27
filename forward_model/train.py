import os
import config

from AEML.data import ADM
from AEML.models.MLP import DukeMLP


def main():
    os.chdir(config.DATA_DIR)

    train_loader, test_loader, test_x, test_y = ADM(
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

    model.train_(
        train_loader,
        test_loader,
        epochs=config.EPOCHS,
        optm='Adam',
        weight_decay=config.WEIGHT_DECAY,
        lr=config.LR,
        lr_scheduler_name='reduce_plateau',
        lr_decay_rate=config.LR_DECAY_RATE,
        eval_step=config.EVAL_STEP,
        stop_threshold=config.STOP_THRESHOLD,
    )

    mse = model.evaluate(test_x, test_y, save_output=False)
    print(f"Test MSE: {mse:.6f}")


if __name__ == '__main__':
    main()
