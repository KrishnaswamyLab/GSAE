import numpy as np

import torch
import torch.utils.data
from torch.nn import functional as F

from gsae.utils import eval_metrics


ROOT_DATA_DIR = 'data/final_data/'


def load_seq3(batch_size=100,gnn=False, subsize=None, lognorm=False):

    train_coeffs = np.load(ROOT_DATA_DIR+"seq_3_train_coeffs.npy")
    train_adjs = np.load(ROOT_DATA_DIR+"seq_3_train_adjs.npy")
    train_energies = np.load(ROOT_DATA_DIR+"seq_3_train_energies.npy")

    test_coeffs = np.load(ROOT_DATA_DIR+"seq_3_test_coeffs.npy")
    test_adjs = np.load(ROOT_DATA_DIR+"seq_3_test_adjs.npy")
    test_energies = np.load(ROOT_DATA_DIR+"seq_3_test_energies.npy")


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
    #################tre
    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)


    train_dataset = torch.utils.data.TensorDataset(*train_tup)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    return train_loader, train_tup, test_tup



def load_seq4(batch_size=100, gnn=False, subsize=None):


    train_coeffs = np.load(ROOT_DATA_DIR+"seq_4_train_coeffs.npy")
    train_adjs = np.load(ROOT_DATA_DIR+"seq_4_train_adjs.npy")
    train_energies = np.load(ROOT_DATA_DIR+"seq_4_train_energies.npy")

    test_coeffs = np.load(ROOT_DATA_DIR+"seq_4_test_coeffs.npy")
    test_adjs = np.load(ROOT_DATA_DIR+"seq_4_test_adjs.npy")
    test_energies = np.load(ROOT_DATA_DIR+"seq_4_test_energies.npy")


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


    train_coeffs = np.load(ROOT_DATA_DIR+"hivtar_train_coeffs.npy")
    train_adjs = np.load(ROOT_DATA_DIR+"hivtar_train_adjs.npy")
    train_energies = np.load(ROOT_DATA_DIR+"hivtar_train_energies.npy")


    test_coeffs = np.load(ROOT_DATA_DIR+"hivtar_test_coeffs.npy")
    test_adjs = np.load(ROOT_DATA_DIR+"hivtar_test_adjs.npy")
    test_energies = np.load(ROOT_DATA_DIR+"hivtar_test_energies.npy")


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

    train_coeffs = np.load(ROOT_DATA_DIR+"teb_train_coeffs.npy")
    train_adjs = np.load(ROOT_DATA_DIR+"teb_train_adjs.npy")
    train_energies = np.load(ROOT_DATA_DIR+"teb_train_energies.npy")
    

    test_coeffs = np.load(ROOT_DATA_DIR+"teb_test_coeffs.npy")
    test_adjs = np.load(ROOT_DATA_DIR+"teb_test_adjs.npy")
    test_energies = np.load(ROOT_DATA_DIR+"teb_test_energies.npy")


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