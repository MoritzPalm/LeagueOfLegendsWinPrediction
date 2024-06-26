import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc

import wandb

wandb.login()

if torch.cuda.is_available():
    print(f'PyTorch version: {torch.__version__}')
    print('*' * 10)
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    torch.cuda.set_device(1)
    print(f'Device Name: {torch.cuda.get_device_name()}')
    print(f'Current device: {torch.cuda.current_device()}')
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
            'values': [128, 256]
        },
        'num_layers': {
            'min': 2,
            'max': 12
        },
        'dropout_prob': {
            'values': [0, 0.1]
        },
        'activation': {
            'values': ['ReLU', 'ELU', 'LeakyReLU']
        },
        'batch_size': {
            'values': [128, 256]
        },
        'learning_rate': {
            'min': 1e-10,
            'max': 1e-5,
            'distribution': 'log_uniform_values'
        },
        'max_epochs': {
            'value': 2000
        },
        'patience': {
            'values': [30]
        },
        'dataset': {
            'values': ['fs_ohc', 'fs_only']
        },
        'optimizer': {
            'values': ['Adam', 'SGD']
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
            #self.linear_relu_stack.append(self.dropout)
            #self.linear_relu_stack.append(nn.BatchNorm1d(self.hidden_size))
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
                 output_size, activation, learning_rate, optimizer):
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
        self.optimizer = optimizer


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
        fpr, tpr, threshold = roc_curve(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        df_fpr = pd.DataFrame(fpr)
        df_tpr = pd.DataFrame(tpr)
        df_threshold = pd.DataFrame(threshold)
        fpr_table = wandb.Table(dataframe=df_fpr)
        tpr_table = wandb.Table(dataframe=df_tpr)
        threshold_table = wandb.Table(dataframe=df_threshold)
        wandb.log({'fpr_table': fpr_table})
        wandb.log({'tpr_table': tpr_table})
        wandb.log({'threshold_table': threshold_table})
        print(f'test_confusion_matrix {self.confusion_matrix(y_hat, y)}')
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            print('Using Adam optimizer')
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.optimizer == 'SGD':
            print('Using SGD')
            optimizer = optim.SGD(self.parameters(), lr= self.hparams.learning_rate)
        else:
            raise ValueError(f'optimizer {self.optimizer} not supported')
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
        data_dir = 'data/static_16_12_23/processed'
        config = wandb.config

        wandb_logger = WandbLogger()
        training_data = wandb.Artifact('training_data', type='dataset')
        training_data.add_dir(data_dir)
        wandb_logger.experiment.log_artifact(training_data)
        if config.dataset == "fs_only":
            data_dir += '/fs_only'
        elif config.dataset == "fs_ohc":
            data_dir += '/fs_ohc'
        else:
            raise ValueError(f'Dataset {config.dataset} not supported')
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

        model = LNN(input_size=input_size, hidden_size=config.hidden_size, num_layers=config.num_layers, output_size=1,
                    dropout_prob=config.dropout_prob, activation=config.activation, learning_rate=config.learning_rate,
                    optimizer=config.optimizer)
        wandb_logger.watch(model)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath='checkpoints/',
                                              filename='nn-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='max', every_n_epochs=1)
        early_stop_callback = EarlyStopping(monitor='val_loss',
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
