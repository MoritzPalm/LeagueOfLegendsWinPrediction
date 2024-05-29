import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb

wandb.login()

config = {
    "learning_rate": 0.1,
    "architecture": "NN",
    "dataset": "static_1.1",
    "epochs": 500,
    "classes": 2,
    "batch_size": 32,
    "num_layers": 2,
    "hidden_size": 128,
    "dropout_prob": 0.3,
    "input_size": 229,
    "output_size": 2,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss",
    "activation": "ReLU",
}


class StaticDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data = torch.tensor(np.load(data_dir)[:, :-1], dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.load(data_dir)[:, -1], dtype=torch.int64, device=device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, 1:]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


def get_train_data(val_split=0.2) -> (Dataset, Dataset):
    """
    Returns a train and validation dataset
    :param val_split: percentage of data to use for validation
    :return: train_dataset, val_dataset
    """
    dataset = StaticDataset('../data/processed/train_static.npy')
    train_len = int(len(dataset) * val_split)
    val_len = len(dataset) - train_len
    print(f'train_len: {train_len}, val_len: {val_len}')
    return torch.utils.data.random_split(dataset, [train_len, val_len])


def get_test_data() -> Dataset:
    """
    Returns a test dataset
    :return: test_dataset
    """
    return StaticDataset('../data/processed/test_static.npy')


def make_loader(dataset, batch_size=64) -> DataLoader:
    """
    Returns a DataLoader for the given dataset
    :param dataset:
    :param batch_size:
    :return:
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, num_classes=2):
        super(NeuralNetwork, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.linear_relu_stack.append(nn.Linear(input_size, hidden_size))
            else:
                self.linear_relu_stack.append(nn.Linear(hidden_size, hidden_size))
            self.linear_relu_stack.append(nn.ReLU())
            self.linear_relu_stack.append(self.dropout)
        self.linear_relu_stack.append(nn.Linear(hidden_size, num_classes))
        self.linear_relu_stack.append(nn.Sigmoid())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(model, train_loader, val_loader, criterion, optimizer, config):
    """
    Trains the given model
    :param model:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param optimizer:
    :param config:
    :return:
    """
    wandb.watch(model, criterion, log="all", log_freq=10)
    wandb_metrics = {}
    loss_vals = []
    vloss_vals = []
    acc_vals = []
    vacc_vals = []
    since = time.time()
    for epoch in tqdm(range(config.epochs)):
        print(f'Epoch {epoch}/{config.epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.
            running_corrects = 0

            for i, (matches, labels) in enumerate(loader):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(matches)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * matches.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            if phase == 'train':
                loss_vals.append(epoch_loss)
                acc_vals.append(epoch_acc)
                wandb_metrics['train_loss'] = epoch_loss
                wandb_metrics['train_acc'] = epoch_acc
            else:
                vloss_vals.append(epoch_loss)
                vacc_vals.append(epoch_acc)
                wandb_metrics['val_loss'] = epoch_loss
                wandb_metrics['val_acc'] = epoch_acc
    wandb.log(wandb_metrics)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def test(model, test_loader):
    """
    Tests the model
    :param model:
    :param test_loader:
    :return:
    """
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for matches, labels in test_loader:
            matches, labels = matches.to(device), labels.to(device)
            outputs = model(matches)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test matches: {correct / total:%}")

        wandb.log({"test_accuracy": correct / total})


def make(config) -> (NeuralNetwork, DataLoader, DataLoader, nn.CrossEntropyLoss, optim.Optimizer):
    """
    Returns a model, train_loader, val_loader, criterion, optimizer
    :param config:
    :return:
    """
    train_dataset, val_dataset = get_train_data()
    train_loader = make_loader(train_dataset, config.batch_size)
    val_loader = make_loader(val_dataset, config.batch_size)
    model = NeuralNetwork(config.input_size, config.hidden_size, config.num_layers, config.dropout_prob,
                          config.output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    return model, train_loader, val_loader, criterion, optimizer


def model_pipeline(hyperparameters):
    with wandb.init(project='leaguify', config=hyperparameters):
        config = wandb.config
        global device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training")

        model, train_loader, val_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, val_loader, criterion, optimizer, config)
        test(model, val_loader)


if __name__ == '__main__':
    model_pipeline(config)
