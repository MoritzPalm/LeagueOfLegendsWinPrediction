import lightning as L
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

import wandb

wandb.login()

if torch.cuda.is_available():
    print(f'PyTorch version: {torch.__version__}')
    print('*' * 10)
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'hidden_size': {
            'values': [128, 256, 512, 1024]
        },
        'num_layers': {
            'min': 2,
            'max': 15
        },
        'dropout_prob': {
            'values': [0.3, 0.4, 0.5]
        },
        'activation': {
            'values': ['ReLU', 'ELU', 'LeakyReLU']
        },
        'batch_size': {
            'values': [128, 256]
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-2
        },
        'max_epochs': {
            'value': 500
        },
        'patience': {
            'values': [30, 40]
        },
        'merged': {
            'values': ["True", "ohc"]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='leaguify')


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob,
                 output_size=1, activation=nn.ReLU(), ):
        super(NeuralNetwork, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers - 1):
            next_hidden_size = self.hidden_size
            self.linear_relu_stack.append(self.dropout)
            self.linear_relu_stack.append(nn.BatchNorm1d(self.hidden_size))
            self.linear_relu_stack.append(nn.Linear(self.hidden_size, next_hidden_size))
            self.linear_relu_stack.append(activation)
            self.hidden_size = next_hidden_size
        self.linear_relu_stack.append(nn.Linear(self.hidden_size, self.output_size))
        self.linear_relu_stack.append(nn.Sigmoid())

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class LNN(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob,
                 output_size=1, activation=nn.ReLU(), learning_rate=1e-3):
        super().__init__()
        if activation == 'ReLU':
            activation = nn.ReLU()
        elif activation == 'ELU':
            activation = nn.ELU()
        elif activation == 'LeakyReLU':
            activation = nn.LeakyReLU()
        else:
            raise ValueError(f'Activation {activation} not supported')
        self.model = NeuralNetwork(input_size, hidden_size, num_layers,
                                   dropout_prob, output_size, activation)
        self.criterion = nn.BCELoss()
        self.save_hyperparameters()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        y_hat = self.model(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        y_hat = self.model(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        y_hat = self.model(x).squeeze(-1)
        print(f'y_shape: {y.shape}, y_hat_shape: {y_hat.shape}')
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('test_f1', self.f1(y_hat, y), prog_bar=True)
        print(f'test_confusion_matrix {self.confusion_matrix(y_hat, y)}')
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class StaticDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data = torch.tensor(np.load(data_dir)[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(np.load(data_dir)[:, -1], dtype=torch.int64)
        self.shape = self.data.shape
        self.transform = transform
        self.target_transform = target_transform
        self.print_statistics()

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

    def print_statistics(self):
        print(f'Number of samples: {len(self.data)}')
        print(f'Number of features: {len(self.data[0])}')
        print(f'Number of labels: {len(self.labels)}')
        print(f'Number of classes: {len(np.unique(self.labels.cpu().numpy()))}')
        print(f'Number of samples per class: {np.bincount(self.labels.cpu().numpy())}')


def main(config=None):
    """

    :param config:
    :return:
    """
    with wandb.init(config=config):
        data_dir = '../../data/static_11_12_23/processed'
        config = wandb.config

        wandb_logger = WandbLogger()
        training_data = wandb.Artifact('training_data', type='dataset')
        training_data.add_dir(data_dir)
        wandb_logger.experiment.log_artifact(training_data)
        if config.merged == "True":
            data_dir += '/merged_only'
            X_train = StaticDataset(data_dir + '/train_static.npy')
            X_val = StaticDataset(data_dir + '/val_static.npy')
            X_test = StaticDataset(data_dir + '/test_static.npy')
            train_loader = DataLoader(X_train,
                                      batch_size=config.batch_size,
                                      shuffle=True)
            val_loader = DataLoader(X_val, batch_size=X_val.__len__(),
                                    shuffle=False)
            test_loader = DataLoader(X_test, batch_size=X_test.__len__(),
                                     shuffle=False)
            input_size = X_train.shape[1] - 1
        elif config.merged == "ohc":
            data_dir += '/merged_ohc'
            X_train = StaticDataset(data_dir + '/train_static.npy')
            X_val = StaticDataset(data_dir + '/val_static.npy')
            X_test = StaticDataset(data_dir + '/test_static.npy')
            train_loader = DataLoader(X_train, batch_size=config.batch_size,
                                      shuffle=True)
            val_loader = DataLoader(X_val, batch_size=X_val.__len__(),
                                    shuffle=False)
            test_loader = DataLoader(X_test, batch_size=X_test.__len__(),
                                     shuffle=False)
            input_size = X_train.shape[1] - 1
        else:
            data_dir += '/default'
            X_train = StaticDataset(data_dir + '/train_static.npy')
            X_val = StaticDataset(data_dir + '/val_static.npy')
            X_test = StaticDataset(data_dir + '/test_static.npy')
            train_loader = DataLoader(X_train, batch_size=config.batch_size,
                                      shuffle=True)
            val_loader = DataLoader(X_val, batch_size=X_val.__len__(),
                                    shuffle=False)
            test_loader = DataLoader(X_test, batch_size=X_test.__len__(),
                                     shuffle=False)
            input_size = X_train.shape[1] - 1

        model = LNN(input_size, config.hidden_size, config.num_layers, config.dropout_prob)
        wandb_logger.watch(model)
        checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                              dirpath='checkpoints/',
                                              filename='nn-{epoch:02d}-{val_acc:.2f}',
                                              save_top_k=3,
                                              mode='max', every_n_epochs=1)
        early_stop_callback = EarlyStopping(monitor='val_acc',
                                            patience=config.patience,
                                            mode='max')
        trainer = L.Trainer(max_epochs=config.max_epochs, accelerator=device,
                            devices=1, logger=wandb_logger, callbacks=
                            [early_stop_callback, checkpoint_callback])
        trainer.test(model, test_loader)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)


if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=1)
