
from tqdm import tqdm
import numpy as np

from numpy import linalg as LA

import torch
from torch.nn import functional as F
from einops import rearrange

import scipy.stats as stats



# create lazy random walk P, with column sum equals 1
def lazy_random_walk(adj_mat):

    N = adj_mat.shape[-1]

    
    adj_mat = torch.Tensor(adj_mat)
    
    P = F.normalize(adj_mat, p=1, dim=-1)
    P = torch.eye(N) + P
    P = 0.5 * P
    
    return P


# create multiple graph wavelets psi with different j's
def graph_wavelet(P, scales=[1,2,3,4]):

    N = P.shape[-1]

    psi = []
    
    for j in [1,2,3,4]:
        j_high = j
        j_low = j-1

        W_d1 = LA.matrix_power(P,2**j_low) - LA.matrix_power(P,2**j_high)
        
        psi.append(W_d1) # shape: S x N x N
    
    return np.array(psi)


def generate_graph_feature(A, ro=None):
    
    # constants
    N = A.shape[-1]

    # use dirac signals if none provided
    if ro == None:
        ro = np.eye(A.shape[0])

    # create lazy random walk matrix
    P = lazy_random_walk(A)

    # create filter bank
    W = graph_wavelet(P)
    S = W.shape[0]
        
    # ################################ 
    # first order transform
    # ################################
    # 4 x 32 x 32
    u1 = np.abs(W)

    # ################################      
    # second order transform
    # ################################

    second_order_list = []
    
    # ent size to 1 x 4 x 32 x 32
    u2 = u1.reshape(1,4,N,N)

    # get outer product of matrices
    # out shape - > # 4 x 4 x N x N 
    second_ord = np.abs(np.einsum("ix ab, jy lm -> xylm", u2, u2))
    

    second_ord = second_ord.reshape(S**2,N,N) 

    u2 = rearrange(second_ord, 'p n m -> n (p m)') # N x S*S*N
    
    # u2 = batch_sec_ord_tran.reshape(N,-1)
    u1 = rearrange(u1, 'a b c -> b (a c)')

    return np.concatenate([u1,u2], -1)


def transform_dataset(folds_array):
    
    scat_coefficients = []
        
    for indx, entry in enumerate(tqdm(folds_array)):
        
        scat_coefficients.append(generate_graph_feature(entry))
        
    return np.array(scat_coefficients)



def get_normalized_moments(scatcoeff_array):
    
    all_norm_scatcoeffs = []
    
    for indx, entry in enumerate(tqdm(scatcoeff_array)):
        
        # entry = N x F
        
        zeroth_order = np.mean(entry,0).reshape(1,-1)
#         print('zero min: {}'.format(zeroth_order.min()))
        
        first_order = np.var(entry,0).reshape(1,-1)
#         print('first_order min: {}'.format(first_order.min()))
        
        skew = stats.skew(entry,bias=0,axis=0).reshape(1,-1)
#         print('skew min: {}'.format(skew.min()))
        
        kurtosis = stats.kurtosis(entry,axis=0).reshape(1,-1)
#         print('kurtosis min: {}'.format(kurtosis.min()))

        normed_coeffs = np.concatenate((zeroth_order,
                                        first_order,
                                        skew, 
                                        kurtosis),1)
        
        all_norm_scatcoeffs.append(normed_coeffs)
    
    return np.array(all_norm_scatcoeffs)
        