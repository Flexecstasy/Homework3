import logging
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from utils.experiment_utils import setup_logging, Timer
from utils.model_utils import FullyConnectedNet, train_model
from utils.visualization_utils import plot_learning_curves
import numpy as np


def get_synthetic_data(n_samples=1000, input_dim=20, num_classes=2):
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(n_samples,)).astype(np.int64)
    return X, y


def run_depth_experiments(input_size, num_classes, depths, epochs, device):
    setup_logging('results/depth_experiments/log.txt')

    # Генерация синтетических данных
    X, y = get_synthetic_data(n_samples=5000, input_dim=input_size, num_classes=num_classes)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=64, shuffle=True),
        'val': DataLoader(val_ds, batch_size=64)
    }

    for depth in depths:
        sizes = [input_size] + [128] * (depth - 1) + [num_classes]
        model = FullyConnectedNet(sizes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logging.info(f"Starting depth {depth}")
        with Timer():
            history = train_model(model, dataloaders, criterion, optimizer, epochs, device)

        plot_learning_curves(history, f'plots/depth_{depth}.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_depth_experiments(args.input_size, args.num_classes, [1,2,3,5,7], args.epochs, args.device)