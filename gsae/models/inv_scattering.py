

from einops import rearrange, 
from einops.layers.torch import Rearrange

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class generator_soft(nn.Module):
    def __init__(self, input_size, hidden_size, graph_size, dev, node_dim):
        super(generator_soft, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_size = graph_size
        self.node_dim = node_dim
        self.dev = dev
        
        
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.nonlin1 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.nonlin2 = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(hidden_size, graph_size*node_dim)
        self.bn3 = nn.BatchNorm1d(graph_size*node_dim)
        self.nonlin3 = nn.LeakyReLU()
        
        self.fc4 = nn.Linear(graph_size*node_dim, graph_size*node_dim)
        
        
        # weight init
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
                
    def innerprod(self, z):
        A = torch.sigmoid(torch.bmm(z,z.transpose(-1,-2)))
        return A

    def forward(self,z):
        
        batch_size = z.shape[0]

        h = self.nonlin1(self.bn1(self.fc1(z)))
        h = self.nonlin2(self.bn2(self.fc2(h)))
        h = self.nonlin3(self.bn3(self.fc3(h)))
        h = self.fc4(h)
        
        h = h.reshape(batch_size, self.graph_size, self.node_dim)       
        
        self_loop = torch.eye(self.graph_size).unsqueeze(0).repeat(batch_size,1,1).to(self.dev)
        
        a_hat = self.innerprod(h)
    
        a_hat = a_hat - self_loop*0.99
              
        return a_hat



     
class generator_nonreg(nn.Module):
    def __init__(self, input_size, hidden_size, graph_size):
        super(generator_nonreg, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_size = graph_size

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.nonlin1 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.nonlin2 = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.nonlin3 = nn.LeakyReLU()
        
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        
        
        # weight init
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        
    def innerprod(self, z):
        A = torch.softmax(torch.bmm(z,z.transpose(-1,-2)),-1)
        return A

    def forward(self,z):
        z = z.reshape(-1, self.input_size)
        h = self.nonlin1(self.fc1(z))
        h = self.nonlin2(self.fc2(h))
        h = self.nonlin3(self.fc3(h))
        h = self.fc4(h)
        
        h = h.reshape(-1, self.graph_size, self.hidden_size)
        a_hat = self.innerprod(h)
        
        return a_hat
    

    
class generator_2h(nn.Module):
    def __init__(self, input_size, hidden_size,graph_size,node_dim=50):
        super(generator_2h, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_size = graph_size
        self.node_dim = node_dim

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.nonlin1 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(hidden_size,node_dim*graph_size)
        self.nonlin2 = nn.LeakyReLU()
        
        self.fc4 = nn.Linear(hidden_size,node_dim*graph_size)

        
        # weight init
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        
    def innerprod(self, z):
        A = torch.softmax(torch.bmm(z,z.transpose(-1,-2)),-1)
        return A

    def forward(self,z):
        z = z.reshape(-1, self.input_size)
        h = self.nonlin1(self.bn1(self.fc1(z)))
        h = self.nonlin2(self.bn2(self.fc2(h)))
        h = self.fc4(h)
        
        h = h.reshape(-1, self.graph_size, self.node_dim)
        
        a_hat = self.innerprod(h)   
        return a_hat

class scattering(nn.Module):
    def __init__(self, dev):
        super(scattering, self).__init__()
        
        self.rearange0 = Rearrange(' a b c d -> a c (b d)')
        self.dev = dev
        
    def lazy_walk(self, adj_mat): 
        
        # constants
        N = adj_mat.size(-1) # num nodes
        B = adj_mat.size(0) # batch size
        
        # lazy walk matrix
        P = F.normalize(adj_mat, p=1, dim=-1)
        P = torch.eye(N).unsqueeze(0).repeat(B,1,1).to(self.dev) + P
        P = 0.5 * P
        
        return P
        
    def gen_wavelet(self, p_batch, scales=[1,2,3,4]):
        
        # constants
        B = p_batch.size()[0] # batch size
        N = p_batch.size()[1] # num nodes
        S = len(scales) # num dyads
        
        # iterate over batch
        batch_psis = []
        for indx, ent in enumerate(p_batch): 
            psi = []
        
            # iterate over scales
            for j in scales:
                j_high = j
                j_low = j-1
                W_i = torch.matrix_power(ent,2**j_low) - torch.matrix_power(ent,2**j_high)
                psi.append(W_i.unsqueeze(0))
            
            # combine the filters into a len(scales) x N x N tensor
            psi = torch.cat(psi,0) # S x N x N
                        
            # collect them into batch list
            batch_psis.append(psi.unsqueeze(0)) # 1 x S x N x N 
        
        # combine filter banks for all entries in batch
        batch_psi = torch.cat(batch_psis, 0) # B x S x N x N
        
        return batch_psi

    def gettransform(self, batch_psi):
        
        # get dimensions
        B, S, N, _ = batch_psi.size()

        # ################################      
        # first order transform
        # ################################
        batch_first_ord = rearrange(batch_psi, 'b s n f -> b n s f ').reshape(B,N,-1)
        batch_first_ord = torch.abs(batch_first_ord)
        
        # ################################      
        # second order transform
        # ################################

        second_order_list = []
        
        for ind, ent in enumerate(batch_psi):
            
            # ent size to 1 x 4 x 32 x 32
            ent = ent.unsqueeze(0)
            ent = torch.abs(ent)
            
            # get outer product of matrices
            # out shape - > # 4 x 4 x N xN 
            
            second_ord = torch.abs(torch.einsum("ix ab, jy lm -> xylm", ent, 
                                      ent)) 
            
            second_ord = second_ord.reshape(1, S**2,N,N) # 1 x S**2 x N xN 
            
            # add to batch list
            second_order_list.append(second_ord)
        
        # create the batch tensor from list
        batch_sec_ord_tran = torch.cat(second_order_list, 0) # B x S**2 x N x N 
    
        # Rearrange(' a b c d -> a c (b d)')
        batch_sec_ord_tran = self.rearange0(batch_sec_ord_tran)
        
        coeffs = torch.cat([batch_first_ord,
                            batch_sec_ord_tran],-1)
        
        return coeffs

    def forward(self, a_hat):
        
        # get batch lazy walk matrices
        batch_p = self.lazy_walk(a_hat)
        
        # get graph wavelets
        batch_w = self.gen_wavelet(batch_p)
        
        # get tranform
        batch_tran = self.gettransform(batch_w)

        return batch_tran



#######################################
# OVERAL MODEL
#######################################


class inv_scatter(nn.Module):
    def __init__(self, input_size, hidden_size, node_dim, dev, graph_size):
        super(inv_scatter, self).__init__()

        self.scatter = scattering(dev)
        self.generate = generator_soft(input_size = input_size,
                                       hidden_size = hidden_size,
                                       graph_size = graph_size, 
                                       node_dim = node_dim,
                                       dev=dev).to(dev)
        
    def forward(self, phi):
        
        # generate A hat
        a_hat = self.generate(phi)
        
        # generate phi_hat 
        phi_hat = self.scatter(a_hat)
        
        return a_hat, phi_hat


def train_inv_model(train_loader, graph_size=32, 
                    hidden_size=500, node_dim=50, 
                    n_epochs=10, lr=0.0001, dev='cpu'):
    
    
    input_size = train_loader.dataset.tensors[0].shape[-1]
    
    batch_size = train_loader.batch_size

    # choose device
    model = inv_scatter(input_size, hidden_size=hidden_size,
                        graph_size=graph_size, node_dim =node_dim, dev=dev).to(dev)

    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    samples = []

    loss_list = []


    for ep in range(n_epochs):

        for i, batch in enumerate(train_loader):
            model.train()

            z_true, a_true = batch
            z_true = z_true.to(dev)
            a_true = a_true.to(dev)
                        
            a_hat, z_hat = model(z_true)
            
            loss = nn.BCELoss()(a_hat,a_true)
            print("epoch: {} \t batch: {}\t loss: {:.3f} ".format(ep,i, loss.item()))

            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            # collect samples
            if i == 0:
                samples.append(a_hat.cpu().detach().numpy())
                

    return samples, loss_list, model.cpu().eval()