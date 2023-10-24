import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb

wandb.login()

# Hyperparameters
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'dataset': 'timeline_1.0',
    'architecture': 'GRU',
    'parameters': {
        'epochs': {
            'values': [50, 100, 200, 500]
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
        },
        'hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'num_layers': {
            'values': [1, 2, 3]
        },
        'dropout': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'bidirectional': {
            'values': [True, False]
        },
        'optimizer': {
            'values': ['Adam', 'SGD']
        },
        'loss_function': {
            'values': ['CrossEntropyLoss', 'BCEWithLogitsLoss']
        },
        'activation': {
            'values': ['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'Tanh']
        },
        'weight_decay': {
            'values': [0.0, 0.0001, 0.001, 0.01, 0.1]
        },
        'gradient_clipping': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'input_size': {
            'value': 381
        },
        'output_size': {
            'value': 2
        },
        'regularization_type': {
            'value': 'l2'
        },
        'early_stopping': {
            'values': [True, False]
        },
        'early_stopping_patience': {
            'values': [5, 10, 15, 20, 25]
        },
        'early_stopping_min_delta': {
            'values': [0.0, 0.0001, 0.001, 0.01, 0.1]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project='leaguify')
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


class TimelineDataset(Dataset):
    """
    Dataset class for the timeline data
    """

    def __init__(self, data_dir, sequence_length, transform=None, target_transform=None):
        self.data = torch.tensor(np.load(data_dir)[:, :-1], dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.load(data_dir)[:, -1], dtype=torch.int64, device=device)
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sample = self.data[idx:idx + self.sequence_length, :]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


def get_train_data(sequence_length=16, val_split=0.2):
    """
    Returns the training and validation data
    :param sequence_length:
    :param val_split:
    :return:
    """
    dataset = TimelineDataset('../data/processed/train_timeline.npy', sequence_length)
    train_len = int(len(dataset) * val_split)
    val_len = len(dataset) - train_len
    return torch.utils.data.random_split(dataset, [train_len, val_len])


def get_test_data(sequence_length=16):
    """
    Returns the test data
    :param sequence_length:
    :return:
    """
    full_dataset = TimelineDataset('../data/processed/test_timeline.npy', sequence_length)
    return full_dataset


def make_loader(dataset, batch_size=64):
    """
    Returns a DataLoader for the given dataset
    :param dataset:
    :param batch_size:
    :return:
    """
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)


class GRU(nn.Module):
    """
    GRU model
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, drop_prob=0.2, activation='relu'):
        super(GRU, self).__init__()
        self.activation = build_activation(activation)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=drop_prob, bidirectional=bidirectional)
        # dropout only on gru layer(s)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        """
        Forward pass
        :param x:
        :param h:
        :return:
        """
        # weights are not re-initialized at each batch
        out, h = self.gru(x, h)
        out = self.fc(self.activation(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state, currently unused
        :param batch_size:
        :return:
        """
        weight = next(self.parameters()).data
        hidden = weight.new(1, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def train(config):
    with (wandb.init(config=config)):
        config = wandb.config
        train_data, val_data = get_train_data()
        train_loader = make_loader(train_data, config.batch_size)
        val_loader = make_loader(val_data, config.batch_size)
        model = GRU(config.input_size, config.hidden_size, config.output_size,
                    config.num_layers, config.bidirectional, config.dropout, config.activation).to(device)
        optimizer = build_optimizer(model, config.optimizer,
                                    config.learning_rate, config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        wandb.watch(model, criterion, log='all', log_freq=10)

        total_batches = len(train_loader) * config.epochs
        example_count = 0
        batch_count = 0
        loss_vals = []
        for epoch in tqdm(range(config.epochs)):
            model.train()
            epoch_loss = []
            for _, (matches, labels) in enumerate(train_loader):
                output, h = model(matches)  # hidden state is not passed to re-init at each batch
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                if config.gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()
                example_count += len(matches)
                batch_count += 1
                if (batch_count + 1) % 25 == 0:
                    train_log(loss, example_count, epoch)
                epoch_loss.append(loss.item() * matches.size(0))
            loss_vals.append(np.mean(epoch_loss))
            val_loss = validate(model, val_loader, criterion)
            wandb.log({"train_loss": np.mean(epoch_loss)})
            wandb.log({"val_loss": val_loss})
            if config.early_stopping:
                if EarlyStopper(config.early_stopping_patience,
                                config.early_stopping_min_delta
                                ).early_stop(val_loss):
                    wandb.log({'has_early_stopped': True})
                    break


def train_log(loss, example_count, epoch):
    """
    Logs the training loss
    :param loss:
    :param example_count:
    :param epoch:
    :return:
    """
    wandb.log({"epoch": epoch, "loss": loss}, step=example_count)
    print(f"Loss after {str(example_count).zfill(5)} examples: {loss:.3f}")


def validate(model, val_loader, criterion) -> float:
    """
    Tests the model
    :param criterion: the loss function
    :param model: the model
    :param val_loader: the test data loader
    :return: the loss averaged over the validation set
    """
    with torch.no_grad():
        correct, total = 0, 0
        y_true = []
        y_pred = []
        running_loss = 0.0
        for matches, labels in val_loader:
            print(f'matches: {matches.shape}, labels: {labels.shape}')
            matches, labels = matches.to(device), labels.to(device)
            model.eval()
            output, h = model(matches)
            loss = criterion(output, labels)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            running_loss += loss.item() * matches.size(0)

        accuracy = correct / total
        f1 = f1_score(y_true, y_pred)
        print(f"Accuracy of the model on the {total} " +
              f"validation matches: {accuracy:%}")
        print(f"F1 score of the model on the {total} " +
              f"validation matches: {f1}")
        wandb.log({"val_accuracy": correct / total})
        wandb.log({"val_f1": f1})
        return running_loss / total


def build_optimizer(network, optimizer, learning_rate, weight_decay) -> optim.Optimizer:
    """
    Builds the optimizer, either SGD or Adam
    :param weight_decay:
    :param network:
    :param optimizer: the optimizer to use (sgd, adam)
    :param learning_rate:
    :return:
    """
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer not supported")
    return optimizer


def build_activation(activation) -> nn.Module:
    """
    Builds the activation function
    :param activation: the activation function to use (relu, leakyrelu, elu, selu, tanh)
    :return: the activation function
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError("Activation function not supported")


class EarlyStopper:
    """
    Early stopping class
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss) -> bool:
        """
        Checks if the model should be stopped early
        :param validation_loss:
        :return:
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    wandb.agent(sweep_id, train, count=5)
