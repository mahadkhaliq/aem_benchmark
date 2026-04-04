import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import config
from AEML.data.loader import normalize_np
#checkpoint
CKPT = os.path.join(os.path.dirname(__file__), 'models', 'MLP', 'adm_mlp', 'best_model_forward.pt')


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(CKPT, map_location=device)
    net.eval()
    return net, device



def load_data():
    os.chdir(config.DATA_DIR)
    train_x = pd.read_csv(os.path.join('ADM', 'data_g.csv'), header=None).astype('float32').values
    test_x  = pd.read_csv(os.path.join('ADM', 'testset', 'test_g.csv'), header=None).astype('float32').values
    test_y  = pd.read_csv(os.path.join('ADM', 'testset', 'test_s.csv'), header=None).astype('float32').values
    _, x_max, x_min = normalize_np(train_x.copy())
    test_x, _, _ = normalize_np(test_x, x_max, x_min)
    return test_x, test_y

def predict(net, device, x):
    with torch.no_grad():
        return net(torch.tensor(x).unsqueeze(0).to(device)).cpu().numpy().squeeze()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help='Test sample index (0 to 5867)')
    args = parser.parse_args()

    net, device = load_model()
    test_x, test_y = load_data()

    idx = args.idx
    x = test_x[idx]
    y_true = test_y[idx]
    y_pred = predict(net, device, x)
    mse = np.mean((y_pred - y_true) ** 2)

    print(f"Sample index: {idx}")
    print(f"Input  (14 geometry params): {np.round(x, 4)}")
    print(f"MSE: {mse:.6f}")

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Ground truth', alpha=0.8)
    plt.plot(y_pred, label='Predicted', alpha=0.8)
    plt.title(f'Sample {idx} MSE: {mse:.5f}')
    plt.xlabel('Frequency index')
    plt.ylabel('Absorptivity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results', f'predict_{idx}.png'), dpi=300)
    plt.show()
    print(f"Plot saved to results/predict_{idx}.png")

if __name__ == '__main__':
    main()