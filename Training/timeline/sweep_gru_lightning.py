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

wandb.login()

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
            'min': 1,
            'max': 3
        },
        'dropout_prob': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, ]
        },
        'activation': {
            'value': 'ReLU'
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'learning_rate': {
            'min': 1e-7,
            'max': 1e-3
        },
        'max_epochs': {
            'value': 500
        },
        'patience': {
            'values': [40, 60, 80, 100]
        },
        'sequence_length': {
            'value': 16
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project='leaguify')


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers, dropout_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, gru_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = nn.Sigmoid()(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(1, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LGRU(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers, learning_rate, dropout_prob):
        super().__init__()
        self.model = GRU(input_dim, hidden_dim, output_dim, gru_layers, dropout_prob=dropout_prob)
        self.criterion = nn.BCELoss()
        self.save_hyperparameters()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        output = self.model(x)[0].squeeze(-1)
        loss = self.criterion(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(output, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        output = self.model(x)[0].squeeze(-1)
        loss = self.criterion(output, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(output, y), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        output = self.model(x)[0].squeeze(-1)
        loss = self.criterion(output, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', self.accuracy(output, y), prog_bar=True, logger=True)
        self.log('test_f1', self.f1(output, y), prog_bar=True, logger=True)
        print(f'test_confusion_matrix {self.confusion_matrix(output, y)}')
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class TimelineDataset(Dataset):
    def __init__(self, data_dir, sequence_length, transform=None, target_transform=None):
        self.data = torch.tensor(np.load(data_dir)[:, :-1], dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.load(data_dir)[:, -1], dtype=torch.int64, device=device)
        self.shape = self.data.shape
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.print_statistics()

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

    def print_statistics(self):
        """

        :return:
        """
        print(f'Number of samples: {len(self.data)}')
        print(f'Number of features: {len(self.data[0])}')
        print(f'Number of labels: {len(self.labels)}')
        print(f'Number of classes: {len(np.unique(self.labels.cpu().numpy()))}')
        print(f'Number of samples per class: {np.bincount(self.labels.cpu().numpy())}')


def train(config=None):
    """
    Trains the model
    :return:
    """
    with wandb.init(config=config):
        data_dir = '../../data/timeline_18_12_23/processed'
        config = wandb.config

        wandblogger = WandbLogger()
        training_data = wandb.Artifact('training_data', type='dataset')
        training_data.add_dir(data_dir)
        wandblogger.experiment.log_artifact(training_data)
        X_train = TimelineDataset(data_dir + '/train_timeline.npy', sequence_length=config.sequence_length)
        X_val = TimelineDataset(data_dir + '/val_timeline.npy', sequence_length=config.sequence_length)
        X_test = TimelineDataset(data_dir + '/test_timeline.npy', sequence_length=config.sequence_length)
        train_loader = DataLoader(X_train, batch_size=config.batch_size, shuffle=False)
        val_loader = DataLoader(X_val, batch_size=X_val.__len__(), shuffle=False)
        test_loader = DataLoader(X_test, batch_size=X_test.__len__(), shuffle=False)
        input_size = X_train.shape[1]
        model = LGRU(input_size, config.hidden_size, 1,
                     config.num_layers, config.learning_rate, config.dropout_prob)
        wandblogger.watch(model)
        checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                              dirpath='checkpoints/',
                                              filename='gru-{epoch:02d}-{val_acc:.2f}',
                                              save_top_k=3,
                                              mode='max', every_n_epochs=1)
        early_stop_callback = EarlyStopping(monitor='val_acc',
                                            patience=config.patience,
                                            mode='max')
        trainer = L.Trainer(max_epochs=config.max_epochs, logger=wandblogger, accelerator=device, devices=1,
                            callbacks=[checkpoint_callback, early_stop_callback])
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)


if __name__ == '__main__':
    wandb.agent(sweep_id, function=train, count=5)
