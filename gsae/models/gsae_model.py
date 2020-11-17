
from argparse import ArgumentParser
import datetime
import os
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, precision_score, recall_score

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from gsae.utils import eval_metrics, load_splits

class GSAE(pl.LightningModule):
    def __init__(self, hparams):
        super(GSAE, self).__init__()
        
        self.hparams = hparams
        
        self.input_dim = hparams.input_dim
        self.bottle_dim = hparams.bottle_dim
        
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
        z, mu, logvar = self.embed(x, adj)
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
        total_batches = self.len_ep * num_epochs

        # loss annealing
        weight = min(1, self.trainer.global_step / total_batches)
        reg_loss = weight * reg_loss
        kl_loss = weight* KLD
        
            
        reg_loss = alpha * reg_loss.mean()
        kl_loss + beta * kl_loss

        total_loss = recon_loss  +  reg_loss + kl_loss

        log_losses = {'train_loss' : total_loss.detach(), 
                    'recon_loss' : recon_loss.detach(),
                    'pred_loss' :reg_loss.detach(),
                    'kl_loss': kl_loss.detach()}
        
        return total_loss, log_losses



    def training_step(self, batch, batch_idx):
        x, y  = batch
        x_hat, y_hat, mu, logvar, z = self(x)

        loss, log_losses = self.loss_multi_GVAE(recon_x=x, x=x_hat, 
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                            alpha=self.hparams.alpha, beta=self.hparams.beta,
                                            batch_idx=batch_idx)
            
        return {'loss': loss, 'log': log_losses}
        
    def prepare_data(self):
        
        _, train_tup, test_tup = eval('load_splits.load_{}()'.format(self.hparams.dataset))

        # train dataset
        train_dataset = torch.utils.data.TensorDataset(*train_tup)

        # get valid set
        train_set, val_set = torch.utils.data.random_split(train_dataset, [55000, 15000])

        # train loader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.hparams.batch_size,
                                        shuffle=True)

        # valid loader 
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=self.hparams.batch_size,
                                        shuffle=False)



        # save to system
        self.len_ep = len(train_loader)

        self.train_data = train_loader
        self.valid_data = valid_loader


    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.valid_data
        
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



if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--input_dim', default=32, type=int)
    parser.add_argument('--dataset', default='seq3', type=str)
    parser.add_argument('--bottle_dim', default=25, type=int)
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)

    cl_args = parser.parse_args()

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # parse params
    args = parser.parse_args()

    # get data
    # train_loader, train_tup, test_tup = eval('load_splits.load_{}(gnn=True)'.format(args.dataset))
    # len_ep = len(train_loader)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir =  args.save_dir + 'GSAE_logs_run_{}_{}/'.format(args.dataset,date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb_logger = WandbLogger(name='run_{}_{}_{}'.format(args.dataset, args.alpha, args.bottle_dim),
                                project='rna_project_GSAE', 
                                log_model=True,
                                save_dir=save_dir)
                                
    wandb_logger.log_hyperparams(cl_args.__dict__)

    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='min'
            )

    # init module
    model = GSAE(hparams=args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.n_epochs,
                                             gpus=args.n_gpus,
                                            callbacks=[early_stop_callback],
                                            logger=wandb_logger)
    trainer.fit(model)

    # 1. save embeddings
    _, train_tup, test_tup = eval('load_splits.load_{}()'.format(args.dataset))

    model = model.cpu()
    model.dev_type = 'cpu'

    with torch.no_grad():
        train_embed = model.embed(train_tup[0])
        test_embed =  model.embed(test_tup[0])

    # save data
    print('saving embeddings')
    np.save(save_dir + "train_embedding.npy" , train_embed.cpu().detach().numpy() )
    np.save(save_dir + "test_embedding.npy" , test_embed.cpu().detach().numpy() )


    # EVALUATION ON TEST SET 
    # energy pred mse

    print("getting test set predictions")
    with torch.no_grad():
        x_recon_test = model(test_tup[0])[0]
        y_pred_test = model.predict_from_data(test_tup[0])


    # print("adj type: {}".format(test_tup[1].flatten().numpy()))
    # print("adj_hat type: {}".format(adj_hat_test.flatten().detach().numpy()))

    recon_test_val = nn.MSELoss()(x_recon_test.flatten(), test_tup[0].flatten())
    pred_test_val = nn.MSELoss()(y_pred_test.flatten(), test_tup[1].flatten())

    print("logging test set metrics")
    wandb_logger.log_metrics({'test_recon_MSE':recon_test_val,
                                'test_pred_MSE': pred_test_val})


    print("gathering eval subsets")
    eval_tup_list = [eval_metrics.compute_subsample([test_embed, test_tup[-1]], 10000)[0] for i in range(8)]
    # trainer.test()
    print("getting smoothness vals")
    embed_eval_array= np.expand_dims(np.array([x[0].numpy() for x in eval_tup_list]),0)
    energy_eval_array= np.array([x[1].numpy() for x in eval_tup_list])

    print('embed_eval_array shape: {}'.format(embed_eval_array.shape))
    print('energy_eval_array shape: {}'.format(energy_eval_array.shape))
    
    energy_smoothness = eval_metrics.eval_over_replicates(embed_eval_array,
                                                            energy_eval_array,
                                                eval_metrics.get_smoothnes_kNN,
                                                [5, 10])[0]

    energy_smoothness = eval_metrics.format_metric(energy_smoothness)


    wandb_logger.log_metrics({'e_smooth_k5_mean':energy_smoothness[0][0],
                                'e_smooth_k10_mean': energy_smoothness[0][1],
                                'e_smooth_k5_std': energy_smoothness[1][0],
                                'e_smooth_k10_std': energy_smoothness[1][1]})

