
from argparse import ArgumentParser
import datetime
import os
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, precision_score, recall_score

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from torch_geometric.nn import DenseGCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from gsae.utils import eval_metrics, load_splits, gnn_modules



class GVAE_H(pl.LightningModule):
    def __init__(self, hparams):
        super(GVAE_H, self).__init__()
        
        self.hparams = hparams
        
        if hparams.pool_type == 'stat':
            # since stat moments x4
            self.prepool_dim = hparams.hidden_dim//4
        else: 
            self.prepool_dim = hparams.hidden_dim
        
        # encoding layers
        if hparams.gnn_type == 'gcn':
            self.gcn_1 = DenseGCNConv(hparams.input_dim, hparams.hidden_dim)
            self.gcn_2 = DenseGCNConv(hparams.hidden_dim, hparams.hidden_dim)
        
            self.gcn_31 = DenseGCNConv(hparams.hidden_dim, self.prepool_dim)
            self.gcn_32 = DenseGCNConv(hparams.hidden_dim, self.prepool_dim)

        elif hparams.gnn_type == 'sage':
            self.gcn_1 = gnn_modules.DenseSAGEConv(hparams.input_dim, hparams.hidden_dim)
            self.gcn_2 = gnn_modules.DenseSAGEConv(hparams.hidden_dim, hparams.hidden_dim)

            self.gcn_31 = gnn_modules.DenseSAGEConv(hparams.hidden_dim, self.prepool_dim)
            self.gcn_32 = gnn_modules.DenseSAGEConv(hparams.hidden_dim, self.prepool_dim)

        self.fc2 = nn.Linear(hparams.hidden_dim, hparams.bottle_dim)


        # decoding layers
        self.fc3 = nn.Linear(hparams.bottle_dim, hparams.node_dim*hparams.input_dim)
        self.fc4 = nn.Linear(hparams.node_dim*hparams.input_dim, hparams.node_dim*hparams.input_dim) 

        # energy prediction
        self.regfc1 = nn.Linear(hparams.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 1)

        # diff pool
        if hparams.pool_type == 'diff':
            self.gcn_diff = DenseGCNConv(hparams.hidden_dim, 1)

        if hparams.n_gpus > 0:
            self.dev_type = 'cuda'

        if hparams.n_gpus == 0:
            self.dev_type = 'cpu'
        
        self.eps = 1e-5

    def pool_op(self, z):
        
        if self.hparams.pool_type == 'sum':
            z_p = z.sum(1)

        if self.hparams.pool_type == 'max':
            z_p = z.max(1)[0]

        if self.hparams.pool_type == 'mean':
            z_p = z.mean(1)

        if self.hparams.pool_type == 'stat':
            z_p = self.get_moments(z)

        return z_p.squeeze()

    def moment(self, x, order=1, dim=1 ):
        
        if order == 1:
            moment = torch.mean(x,dim)

        elif order == 2:
            moment = torch.var(x,dim)

        elif order == 3: 
            mean = torch.mean(x, dim, keepdim=True)

            moment = (x - mean)**order
            moment = moment.sum(dim)
            moment = moment/ ( (x.var(dim)**order) + self.eps)
            moment = moment / x.shape[dim] 


        elif order == 4: 
            mean = torch.mean(x, dim, keepdim=True)
            moment = (x - mean)**order
            moment = moment.sum(dim)
            moment = moment/ ((x.var(dim)**2) + self.eps)
            moment = moment / x.shape[dim]
            moment -= 3

        return moment

    def get_moments(self, z):

        z.reshape(z.shape[0], -1)
        z1 = self.moment(z,1)
        z2 = self.moment(z,2)
        z3 = self.moment(z,3)
        z4 = self.moment(z,4)
        
        return torch.cat((z1,z2,z3,z4), axis=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def embed(self, x, adj):

        # node level
        h = F.relu(self.gcn_1(x, adj))
        h = F.relu(self.gcn_2(h, adj))
        mu, logvar = self.gcn_31(h, adj), self.gcn_32(h, adj)
        h = self.reparameterize(mu, logvar)

        if self.hparams.pool_type == 'diff':
            # basically attention via gcn
            s = self.gcn_diff(h, adj)
            s = nn.Softmax(1)(s)

            # graph level
            z = torch.bmm(s.transpose(-1,-2), h).squeeze()

        else:      
            # graph level
            z = self.pool_op(h)

        z = self.fc2(z)

        return z, mu, logvar

    def innerprod(self, z):
        A = torch.sigmoid(torch.bmm(z,z.transpose(-1,-2)))
        return A  

    def decode(self, z):

        h = F.relu(self.fc3(z))
        h = self.fc4(h)

        h = h.reshape(-1, 
                        self.hparams.input_dim, 
                            self.hparams.node_dim)  

        batch_dim = h.size()[0]     
        
        self_loop = torch.eye(self.hparams.input_dim, device=self.dev_type).unsqueeze(0).repeat(batch_dim,1,1)
        
        a_hat = self.innerprod(h)
   
        a_hat = a_hat - self_loop*0.99

        a_hat = F.relu(a_hat)
              
        return a_hat


    def forward(self, x, adj):
        # encoding
        z, mu, logvar = self.embed(x, adj)
        # predict
        y_pred = self.predict(z)
        # recon
        x_hat = self.decode(z)

        return x_hat, y_pred, mu, logvar
        
    def predict(self,z):
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred
    
    def predict_from_data(self,x, adj):
        z = self.embed(x, adj)
        pred = self.predict(z)
        return pred


    def loss_multi_GVAE(self, 
                        recon_x, x,  
                        mu, logvar,
                        y_pred, y, 
                        alpha, beta, batch_idx):

        # reconstruction loss
        recon_loss = nn.BCELoss()(recon_x.flatten(), x.flatten()) 
        
        # regression loss
        reg_loss = nn.MSELoss()(y_pred.reshape(-1), y.reshape(-1)) 
    
        # kl divergence 
        KLD = gnn_modules.kl_div(mu, logvar)

        num_epochs = self.hparams.max_epochs - 5
        total_batches = self.len_ep * num_epochs

        # loss annealing
        weight = min(1, self.trainer.global_step /total_batches)
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
        x, adj, y  = batch
        adj_hat, y_hat, mu, logvar = self(x,adj)

        loss, log_losses = self.loss_multi_GVAE(recon_x=adj_hat,x=adj, 
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                            alpha=self.hparams.alpha, beta=self.hparams.beta,
                                            batch_idx=batch_idx)
            
        return {'loss': loss, 'log': log_losses}
        
    def prepare_data(self):
        
        
        _, train_tup, test_tup = eval('load_splits.load_{}(gnn=True)'.format(self.hparams.dataset))

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
        x, adj, y  = batch

        adj_hat, y_hat, mu, logvar = self(x, adj)

        # reconstruction loss
        recon_loss = nn.BCELoss()(adj_hat.flatten(), adj.flatten()) 
        
        # regression loss
        reg_loss = nn.MSELoss()(y_hat.reshape(-1), y.reshape(-1)) 

        # kl loss
        kl_loss = gnn_modules.kl_div(mu, logvar)
    
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
    parser.add_argument('--node_dim', default=50, type=int)
    parser.add_argument('--dataset', default='seq3', type=str)
    parser.add_argument('--gnn_type', default='gcn', type=str)
    parser.add_argument('--pool_type', default='stat', type=str)
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
    save_dir =  args.save_dir + 'logs_run_{}_{}_{}/'.format(args.dataset, args.pool_type, date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb_logger = WandbLogger(name='run_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.pool_type, args.gnn_type, args.alpha, args.bottle_dim, date_suffix),
                                project='rna_project_GVAE', 
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
    model = GVAE_H(hparams=args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.n_epochs,
                                             gpus=args.n_gpus,
                                           callbacks=[early_stop_callback],
                                            logger=wandb_logger)
    trainer.fit(model)


    # 1. save embeddings
    _, train_tup, test_tup = eval('load_splits.load_{}(gnn=True)'.format(args.dataset))

    model = model.cpu()
    model.dev_type = 'cpu'
    with torch.no_grad():
        train_embed = model.embed(train_tup[0], train_tup[1])[0] 
        test_embed =  model.embed(test_tup[0], test_tup[1])[0]

    # save data
    print('saving embeddings')
    np.save(save_dir + "train_embedding.npy" , train_embed.cpu().detach().numpy() )
    np.save(save_dir + "test_embedding.npy" , test_embed.cpu().detach().numpy() )


    # EVALUATION ON TEST SET 
    # energy pred mse

    print("getting test set predictions")
    with torch.no_grad():
        adj_hat_test, y_pred_test, _,_ = model(test_tup[0], test_tup[1])

    # print("adj type: {}".format(test_tup[1].flatten().numpy()))
    # print("adj_hat type: {}".format(adj_hat_test.flatten().detach().numpy()))

    ap_test = average_precision_score(test_tup[1].flatten().numpy(),
                                    adj_hat_test.flatten().detach().numpy())

    pred_test = nn.MSELoss()(y_pred_test.flatten(), test_tup[-1])

    print("logging test set metrics")
    wandb_logger.log_metrics({'test_energy_mse':pred_test.numpy(),
                                'test_AP': ap_test})


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

