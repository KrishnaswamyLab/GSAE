import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import math
from tqdm import tqdm
import os
import glob
import random

from numpy import linalg as LA
from numpy.random import choice
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import distance as scidist


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from torch.nn import Parameter 


##############################
# GENERAL UTILS
###############################

def compute_subsample(data, subsize=10000):
    """returns subsampled data 

    Args:
        data ([type]): list of numpy arrays or tensors
        subsize (int, optional): size of sub sample

    Returns:
        (list of subsampled data, sub indies)
    """


    assert type(data) in [list, tuple]
    
    sub_inds = choice(np.arange(data[0].shape[0]), subsize, replace=False)

    sub_data = [x[sub_inds] for x in data]

    return sub_data, sub_inds


def plot_loss_curves_VAE(loss_array):
    plt.figure(figsize=(8,4))
    plt.style.use('seaborn-deep')

    plt.plot(np.arange(loss_array.shape[0]), loss_array[:,0], label='recon')
    plt.plot(np.arange(loss_array.shape[0]), loss_array[:,1], label = 'smooth')
    plt.plot(np.arange(loss_array.shape[0]), loss_array[:,2], label='KL')
    plt.title('loss curves')
    plt.legend()
    
    plt.show()


def plot_loss_curves_AE(loss_array):
    plt.figure(figsize=(8,4))
    plt.style.use('seaborn-deep')

    plt.plot(np.arange(loss_array.shape[0]), loss_array[:,0], label='recon')
    plt.plot(np.arange(loss_array.shape[0]), loss_array[:,1], label = 'smooth')
    plt.title('loss curves')
    plt.legend()
    
    plt.show()



def format_metric(array):
    
    array = array.reshape(-1,2)
    
    mean_array = np.round(array.mean(0),3)
    std_array = np.round(array.std(0),3)
    

    return mean_array, std_array 


########################################
# SMOOTHNESS FUNCTIONS
########################################

def get_smoothnes_kNN(embeddings, energies, K):
    """ kNN based graph for smoothness calc

    Args:
        embeddings ([type]): [description]
        energies ([type]): [description]
        K ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    N = embeddings.shape[0]
    
    energies = energies.reshape(N,1)

    # get kNN graph
    print("getting kNN graph")
    A_mat = kneighbors_graph(embeddings, n_neighbors=K, 
                                  mode='connectivity').todense()

    # make symmetric
    A_mat = A_mat + A_mat.T
    A_mat = A_mat.clip(0,1)                         

    
    nn_graph = nx.from_numpy_array(A_mat)
    

    print("computing combinatorial graph laplacian")
    L_mat = nx.laplacian_matrix(nn_graph).todense()
            
    # compute smoothness index
    print("computing smoothness value")
    lap_smooth = np.matmul(L_mat, energies)
    lap_smooth = np.matmul(energies.T, lap_smooth)
    signal_dot = np.matmul(energies.T, energies)
    lap_smooth = lap_smooth/signal_dot
    
    print("smoothness for K={}: {}".format(K, lap_smooth.item()))
    
    return lap_smooth.item()


###################################
# GET SMOOTH VALS 
###################################

def get_smoothness_over_vals(embeddings, energies, smoothness_fn, vals):
    
    smooth_list = []
    for val in vals:
        smooth_indxs = smoothness_fn(embeddings, energies, val)
        smooth_list.append(smooth_indxs)

    return np.array(smooth_list)


def eval_over_replicates(embeddings_reps, energy_reps, 
                        smoothness_fn, val_range):
    """returns an N x M array of smoothness values

    N = number of replicates
    M = number of gamma values used

    Args:
        embeddings_reps ([type]): [description]
        energy_reps ([type]): [description]
        smoothness_fn ([type]): [description]
        val_range ([type]): [description]

    Returns:
        [type]: [description]
    """

    # get dimensions
    n_reps = embeddings_reps.shape[0]
    n_gamma_vals =  embeddings_reps.shape[1]
    n_batch = embeddings_reps.shape[2]
    zdim = embeddings_reps.shape[3]

    eval_rep_embeds = embeddings_reps.reshape(-1,n_batch,zdim)
    print("reshaped eval embedding shape: {}".format(eval_rep_embeds.shape))

    smooth_list = []
    for indx, embed_rep in enumerate(eval_rep_embeds):
        print(indx)

        i = indx // n_gamma_vals

        print("getting smoothness val for replicate: {}".format(i))
        smooth_indxs = []

        for val_i in val_range:
            smooth_val = smoothness_fn(embed_rep,
                                       energy_reps[i],
                                       val_i)
            smooth_indxs.append(smooth_val)

        smooth_list.append(smooth_indxs)
    
    smooth_val_array = np.array(smooth_list)
    

    return np.reshape(smooth_val_array, (n_reps, n_gamma_vals, len(val_range)))

    
###########################
# VIZ UTILS
##########################
def plot_gamma_error_plots(gamma_values, smooth_means, smooth_std, hyp_vals, param='k'):

    fig, ax = plt.subplots(figsize=(7,4))
    
    for indx, val in enumerate(hyp_vals):
        ax.errorbar(gamma_values, smooth_means[:,indx] ,
                     smooth_std[:,indx], fmt='-o', label='{}={}'.format(param,val))

    ax.set_xlabel('gamma value')
    ax.set_ylabel('smoothness index')

    plt.legend()
    
    plt.show()
    

def plot_gamma_single_error(gamma_values, error_means, error_stds, xlabel='gamma val', ylabel='MSE prediction'):

    fig, ax = plt.subplots(figsize=(7,4))

    plt.errorbar(gamma_values, error_means , error_stds, fmt='-o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.show()
    


#######################
# OTHER METRICS
#######################


def get_metrics(y_pred,y_true):
    
    print(y_pred.shape)
    print(y_true.shape)
    
    roc_auc_met = roc_auc_score(y_true,y_pred)
    
    acc_score = accuracy_score( y_true,y_pred.round())
    
    avgprec = average_precision_score(y_true, y_pred)
    
    prec = precision_score(y_true, y_pred.round())
    recall = recall_score(y_true, y_pred.round())
    
    
    return [roc_auc_met,acc_score,avgprec,prec,recall] 