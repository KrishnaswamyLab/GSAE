

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import DenseGCNConv, SAGEConv


from gsae.utils import  gnn_modules



class GVAE(pl.LightningModule):
    def __init__(self, hparams):
        super(GVAE, self).__init__()
        
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
