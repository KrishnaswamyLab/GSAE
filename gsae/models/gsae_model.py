

import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl

class GSAE(pl.LightningModule):
    def __init__(self, hparams):
        super(GSAE, self).__init__()

        self.hparams = hparams

        self.input_dim = hparams.input_dim
        self.bottle_dim = hparams.bottle_dim


        self.fc11 = nn.Linear(self.input_dim, 400)
        self.bn11 = nn.BatchNorm1d(400)

        self.fc12 = nn.Linear(400, 400)
        self.bn12 = nn.BatchNorm1d(400)

        self.fc21 = nn.Linear(400, self.bottle_dim)
        self.fc22 = nn.Linear(400, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, 400)
        self.fc4 = nn.Linear(400, self.input_dim)
        # energy prediction
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 1)


        if hparams.n_gpus > 0:
            self.dev_type = 'cuda'

        if hparams.n_gpus == 0:
            self.dev_type = 'cpu'

        self.eps = 1e-5


    def kl_div(self,mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return KLD

    # main model functions
    def encode(self, x):
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def embed(self, x):
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def predict(self,z):
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred

    def predict_from_data(self,x):
        z = self.embed(x)[0]
        pred = self.predict(z)
        return pred

    def forward(self, x):
        # encoding
        z, mu, logvar = self.embed(x)
        # predict
        y_pred = self.predict(z)
        # recon
        x_hat = self.decode(z)

        return x_hat, y_pred, mu, logvar, z

    def loss_multi_GSAE(self,
                        recon_x, x,
                        mu, logvar,
                        y_pred, y,
                        alpha, beta, batch_idx):

        # reconstruction loss
        recon_loss = nn.MSELoss()(recon_x.flatten(), x.flatten())

        # regression loss
        reg_loss = nn.MSELoss()(y_pred.reshape(-1), y.reshape(-1))

        # kl divergence
        KLD = self.kl_div(mu, logvar)

        num_epochs = self.hparams.max_epochs - 5
        total_batches = self.hparams.len_epoch * num_epochs

        # loss annealing
        weight = min(1, self.trainer.global_step / total_batches)
        reg_loss = weight * reg_loss
        kl_loss = weight* KLD


        reg_loss = alpha * reg_loss.mean()
        kl_loss =  beta * kl_loss

        total_loss = recon_loss  +  reg_loss + kl_loss

        log_losses = {'train_loss' : total_loss.detach(),
                    'recon_loss' : recon_loss.detach(),
                    'pred_loss' :reg_loss.detach(),
                    'kl_loss': kl_loss.detach()}

        return total_loss, log_losses



    def training_step(self, batch, batch_idx):
        x, y  = batch
        x_hat, y_hat, mu, logvar, z = self(x)

        loss, log_losses = self.loss_multi_GSAE(recon_x=x, x=x_hat,
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                            alpha=self.hparams.alpha, beta=self.hparams.beta,
                                            batch_idx=batch_idx)

        return {'loss': loss, 'log': log_losses}

    def validation_step(self, batch, batch_idx):
        x, y  = batch

        x_hat, y_hat, mu, logvar,z = self(x)

        # reconstruction loss
        recon_loss = nn.MSELoss()(x_hat.flatten(), x.flatten())

        # regression loss
        reg_loss = nn.MSELoss()(y_hat.reshape(-1), y.reshape(-1))

        # kl loss
        kl_loss = self.kl_div(mu, logvar)

        total_loss = recon_loss  +  reg_loss + kl_loss

        log_losses = {'val_loss' : total_loss.detach(),
                    'val_recon_loss' : recon_loss.detach(),
                    'val_pred_loss' :reg_loss.detach(),
                    'val_kl_loss': kl_loss.detach()}

        return log_losses

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_reconloss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
        avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss,
                            'val_avg_recon_loss': avg_reconloss,
                            'val_avg_pred_loss':avg_regloss,
                            'val_avg_kl_loss':avg_klloss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

