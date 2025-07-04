import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from utils.experiment_utils import setup_logging, Timer
from utils.model_utils import FullyConnectedNet, train_model
from utils.visualization_utils import plot_learning_curves
import numpy as np


def get_synthetic_data(n_samples=5000, input_dim=20, num_classes=2):
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(n_samples,)).astype(np.int64)
    return X, y


def run_regularization_experiments(input_size, num_classes, epochs, device):
    setup_logging('results/regularization_experiments/log.txt')

    X, y = get_synthetic_data(input_dim=input_size, num_classes=num_classes)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=64, shuffle=True),
        'val': DataLoader(val_ds, batch_size=64)
    }

    configs = [
        {'dropout': None, 'batchnorm': False, 'weight_decay': 0},
        {'dropout': 0.1, 'batchnorm': False, 'weight_decay': 0},
        {'dropout': 0.3, 'batchnorm': False, 'weight_decay': 0},
        {'dropout': 0.5, 'batchnorm': False, 'weight_decay': 0},
        {'dropout': None, 'batchnorm': True, 'weight_decay': 0},
        {'dropout': 0.3, 'batchnorm': True, 'weight_decay': 0},
        {'dropout': None, 'batchnorm': False, 'weight_decay': 1e-4},
    ]

    for cfg in configs:
        label = f"do{cfg['dropout']}_bn{cfg['batchnorm']}_wd{cfg['weight_decay']}"
        logging.info(f"Config {label}")
        sizes = [input_size, 128, 128, num_classes]
        model = FullyConnectedNet(sizes, dropout=cfg['dropout'], batchnorm=cfg['batchnorm'])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg['weight_decay'])

        with Timer():
            history = train_model(model, dataloaders, criterion, optimizer, epochs, device)
        plot_learning_curves(history, f'plots/reg_{label}.png')

    # Адаптивная регуляризация: Dropout уменьшается по эпохам
    model = FullyConnectedNet([input_size,128,128,num_classes], dropout=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs+1):
        # динамически менять dropout
        model.net = nn.Sequential(*[
            layer if not isinstance(layer, torch.nn.Dropout) else torch.nn.Dropout(0.5 * (1 - epoch/epochs))
            for layer in model.net
        ])
        train_model(model, dataloaders, criterion, optimizer, 1, device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_regularization_experiments(args.input_size, args.num_classes, args.epochs, args.device)