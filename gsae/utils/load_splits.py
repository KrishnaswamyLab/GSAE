import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from gsae.utils import eval_metrics

# DEFINE THESE GLOBAL VARIABLES WITH YOUR OWN PATHS TO DATA
########################################
SEQ3_DATA_DIR = ''
SEQ4_DATA_DIR = ''
HIVTAR_DATA_DIR = ''
TEBOWN_DATA_DIR = ''
########################################

def load_seq3(batch_size=100,gnn=False, subsize=None, lognorm=False):

    train_coeffs = np.load(SEQ3_DATA_DIR+"train_normcoeffs_0523.npy")
    train_adjs = np.load(SEQ3_DATA_DIR+"train_adjs_0523.npy")
    train_energies = np.load(SEQ3_DATA_DIR+"train_energies_0523.npy")

    test_coeffs = np.load(SEQ3_DATA_DIR+"test_normcoeffs_0523.npy")
    test_adjs = np.load(SEQ3_DATA_DIR+"test_adjs_0523.npy")
    test_energies = np.load(SEQ3_DATA_DIR+"test_energies_0523.npy")


    if lognorm:
        # shift
        train_coeffs +=  np.abs(train_coeffs.min()) + 1
        test_coeffs += np.abs(train_coeffs.min()) + 1
        
        # log
        train_coeffs = np.log(train_coeffs)
        test_coeffs = np.log(test_coeffs)


    if gnn:
        train_diracs = torch.eye(train_adjs.shape[-1]).unsqueeze(0).repeat(train_adjs.shape[0],1,1)
        train_tup = (torch.Tensor(train_diracs),
                    torch.Tensor(train_adjs),
                    torch.Tensor(train_energies))
    else:
        train_tup = (torch.Tensor(train_coeffs),
                    torch.Tensor(train_energies))



    if gnn:
        test_diracs = torch.eye(test_adjs.shape[-1]).unsqueeze(0).repeat(test_adjs.shape[0],1,1)
        test_tup = (torch.Tensor(test_diracs),
                    torch.Tensor(test_adjs),
                    torch.Tensor(test_energies))

    else:
        test_tup = (torch.Tensor(test_coeffs), 
                    torch.Tensor(test_adjs), 
                    torch.Tensor(test_energies))


    #################
    # SUBSET DATA 
    #################
    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)


    train_dataset = torch.utils.data.TensorDataset(*train_tup)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    return train_loader, train_tup, test_tup



def load_seq4(batch_size=100, gnn=False, subsize=None):

    train_coeffs = np.load(SEQ4_DATA_DIR+"train_normcoeffs_0523.npy")
    train_adjs = np.load(SEQ4_DATA_DIR+"train_adjs_0523.npy")
    train_energies = np.load(SEQ4_DATA_DIR+"train_energies_0523.npy")

    test_coeffs = np.load(SEQ4_DATA_DIR+"test_normcoeffs_0523.npy")
    test_adjs = np.load(SEQ4_DATA_DIR+"test_adjs_0523.npy")
    test_energies = np.load(SEQ4_DATA_DIR+"test_energies_0523.npy")


    if gnn: 
        train_diracs = torch.eye(train_adjs.shape[-1]).unsqueeze(0).repeat(train_adjs.shape[0],1,1)
        train_tup = (torch.Tensor(train_diracs),
                    torch.Tensor(train_adjs),
                    torch.Tensor(train_energies))
    else:
        train_tup = (torch.Tensor(train_coeffs),
                    torch.Tensor(train_energies))


    if gnn:
        test_diracs = torch.eye(test_adjs.shape[-1]).unsqueeze(0).repeat(test_adjs.shape[0],1,1)
        test_tup = (torch.Tensor(test_diracs),
                    torch.Tensor(test_adjs),
                    torch.Tensor(test_energies))

    else:
        test_tup = (torch.Tensor(test_coeffs), 
                    torch.Tensor(test_adjs), 
                    torch.Tensor(test_energies))


    #################
    # SUBSET DATA 
    #################

    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)


    train_dataset = torch.utils.data.TensorDataset(*train_tup)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    return train_loader, train_tup, test_tup


def load_hivtar(batch_size=100,gnn=False,subsize=None):


    train_coeffs = np.load(HIVTAR_DATA_DIR+"train_normcoeffs_0523.npy")
    train_adjs = np.load(HIVTAR_DATA_DIR+"train_adjs_0523.npy")
    train_energies = np.load(HIVTAR_DATA_DIR+"train_energies_0523.npy")


    test_coeffs = np.load(HIVTAR_DATA_DIR+"test_normcoeffs_0523.npy")
    test_adjs = np.load(HIVTAR_DATA_DIR+"test_adjs_0523.npy")
    test_energies = np.load(HIVTAR_DATA_DIR+"test_energies_0523.npy")


    if gnn: 
        train_diracs = torch.eye(train_adjs.shape[-1]).unsqueeze(0).repeat(train_adjs.shape[0],1,1)
        train_tup = (torch.Tensor(train_diracs),
                    torch.Tensor(train_adjs),
                    torch.Tensor(train_energies))
    else:
        train_tup = (torch.Tensor(train_coeffs),
                    torch.Tensor(train_energies))



    if gnn:
        test_diracs = torch.eye(test_adjs.shape[-1]).unsqueeze(0).repeat(test_adjs.shape[0],1,1)
        test_tup = (torch.Tensor(test_diracs),
                    torch.Tensor(test_adjs),
                    torch.Tensor(test_energies))

    else:
        test_tup = (torch.Tensor(test_coeffs), 
                    torch.Tensor(test_adjs), 
                    torch.Tensor(test_energies))


    #################
    # SUBSET DATA 
    #################

    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)



    train_dataset = torch.utils.data.TensorDataset(*train_tup)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)


    return train_loader, train_tup, test_tup
    

def load_tebown(batch_size=100,gnn=False, subsize=None):
    
    train_coeffs = np.load(TEBOWN_DATA_DIR+"train_normcoeffs_0523.npy")
    train_adjs = np.load(TEBOWN_DATA_DIR+"train_adjs_0523.npy")
    train_energies = np.load(TEBOWN_DATA_DIR+"train_energies_0523.npy")
    

    test_coeffs = np.load(TEBOWN_DATA_DIR+"test_normcoeffs_0523.npy")
    test_adjs = np.load(TEBOWN_DATA_DIR+"test_adjs_0523.npy")
    test_energies = np.load(TEBOWN_DATA_DIR+"test_energies_0523.npy")


    if gnn: 
        train_diracs = torch.eye(train_adjs.shape[-1]).unsqueeze(0).repeat(train_adjs.shape[0],1,1)
        train_tup = (torch.Tensor(train_diracs),
                    torch.Tensor(train_adjs),
                    torch.Tensor(train_energies))
    else:
        train_tup = (torch.Tensor(train_coeffs),
                    torch.Tensor(train_energies))

    
    if gnn:
        test_diracs = torch.eye(test_adjs.shape[-1]).unsqueeze(0).repeat(test_adjs.shape[0],1,1)
        test_tup = (torch.Tensor(test_diracs),
                    torch.Tensor(test_adjs),
                    torch.Tensor(test_energies))

    else:
        test_tup = (torch.Tensor(test_coeffs), 
                    torch.Tensor(test_adjs), 
                    torch.Tensor(test_energies))

    #################
    # SUBSET DATA 
    #################
    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)

    train_dataset = torch.utils.data.TensorDataset(*train_tup)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    return train_loader, train_tup, test_tup