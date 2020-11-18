"""
data processing utilities for working with RNA
structure ensembles
"""
import numpy as np
import networkx as nx
from tqdm import tqdm
import re


import math
import os
import glob
import random

from numpy import linalg as LA
from numpy.random import choice
import scipy




#############################################
# Loading raw data
#############################################

def load_from_txt(filename):
    """loads RNA sequence and dotbracket strings from RNAfold txt file

    text file format:
    -------------
    ACUCGGUAGCUAGCUAGCUAGCUAGCUAGCUGAC
    )()()((((((()).....(()())))))) -130.12 3e-10
    )()()(..((((()).....(()()))))) -133.14 3e-11
    ...
    
    Args:
        filename (str): file to be loaded
    
    Returns:
        seq [str]: RNA sequence
        db_list [list]: list of dotbracket strings
        energies [list]: list of structure energies

    """
    print("data loading v2")
    db_list = []
    energies = []

    i = 0
    with open(filename,'r') as f:
        print("parsing file for structures and energies")
        for indx, line in enumerate(f):
            line = line.strip()
            line = re.split('\s+',line)

            if indx == 0:
                seq = str(line[0])
            else:
                db_list.append(line[0])
                energies.append(float(line[1]))

        print("finished!")

    return seq, db_list, energies



def dot2adj(db_str):
    """converts DotBracket str to np adj matrix
    
    Args:
        db_str (str): N-len dot bracket string
    
    Returns:
        [np array]: NxN adjacency matrix
    """
    
    dim = len(str(db_str))

    # get pair tuples
    pair_list = dot2pairs(db_str)
    sym_pairs = symmetrized_edges(pair_list)


    # initialize the NxN mat (N=len of RNA str)
    adj_mat = np.zeros((dim,dim))

    adj_mat[sym_pairs[0,:], sym_pairs[1,:]] = 1
    
    return adj_mat


def dot2pairs(db_str):
    """converts a DotBracket str to adj matrix

    uses a dual-checking method
    - str1 = original str
    - str2 = reversed str

    iterates through both strings simult and collects indices
    forward str iteration: collecting opening indicies - list1
    backwards str iteration: collecting closing indices - list2
    - as soon as a "(" is found in str2, the first entry of list1 is paired
      with the newly added index/entry of list2 
    
    Args:
        dotbracket_str (str): dot bracket string (ex. "((..))")
    
    Returns:
        [array]: numpy adjacency matrix
    """ 
    dim = len(str(db_str))

    # backwards str
    
    rep = {"(": ")", ")": "("} # define desired replacements here

    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    
    #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
    pattern = re.compile("|".join(rep.keys()))
    
#     text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    # pairing indices lists
    l1_indcs = []
    l2_indcs = []
    pair_list = []


    for indx in range(dim):
        
        # checking stage
        # forward str
        if db_str[indx] == "(":

            l1_indcs.append(indx)
 
        if db_str[indx] == ")":
            l2_indcs.append(indx)

        # pairing stage
        # check that either list is not empty
        if len(l2_indcs) * len(l1_indcs) > 0:
            pair = (l1_indcs[-1], l2_indcs[0])
            pair_list.append(pair)
#             print("pair: {}".format(pair))
        
            # cleaning stage
            l1_indcs.pop(-1)
            l2_indcs.pop(0)
        
    
    # get path graph pairs
    G = nx.path_graph(dim)
    path_graph_pairs = G.edges()
    return pair_list + list(path_graph_pairs)


def symmetrized_edges(pairs_list):

    # conver pairs to numpy array [2,-1]
    edge_array = np.array(pairs_list)
 
    # concatenate with opposite direction edges
    # print(edge_array.T[[1,0]].T.shape)
    reverse_edges = np.copy(edge_array)
    reverse_edges[:, [0,1]] = reverse_edges[:, [1,0]]
    full_edge_array = np.vstack((edge_array, reverse_edges))
    return full_edge_array.T


#############################################
# Loading data function 
#############################################

def load_data_into_arrays(file, adj_list=False):
    """loading NRAsubopt output to numpy arrays
    
    Args:
        file ([type]): [description]
        adj_list (bool, optional): [description]. Defaults to True.
    
    Returns:
        [type]: [description]
    """    
    seq, folds, energies = load_from_txt(file)
    
    energy_array = np.array(energies)
    # convert dotbrackets to adj array 
    total_fold_array = np.zeros((len(folds), len(folds[0]), len(folds[0])))
    for j, fold in enumerate(tqdm(folds)): 

        A_mat = dot2adj(fold)
        total_fold_array[j] = A_mat
    
    if adj_list:
        adjs = [x for x in total_fold_array]
        
        return seq, adjs, energy_array
    else:
        return seq, total_fold_array, energy_array



####################################
# CLASS/HAIRPIN IDENTIFYING UTILS
####################################

def hairpin2array(pair_array, graph_size, nx_graph=True):

    adj_array = np.zeros((graph_size, graph_size))

    adj_array[pair_array[0,:], pair_array[1,:]] = 1

    hairpin_adj = adj_array + nx.adjacency_matrix(nx.path_graph(graph_size)).todense()

    output = [hairpin_adj]

    if nx_graph:
        g = nx.from_numpy_array(hairpin_adj)
        output.append(g)
    
    return output


def hairpin_search(dataset, list_hairpin_pairs, threshold):
    """search dataset for hairpins and return labels
    
    Args:
        hairpin_arrays ([type]): K x graph_size x graph_size array
        threshold ([type]): integer threshold - number of matches needed for label
    """    

    N = dataset.shape[0]
    K = len(list_hairpin_pairs)

    label_array = np.zeros((N, K + 1))

    # iterate through dataset
    for indx, adj in enumerate(tqdm(dataset)):

        for h_cls in range(K):

            num_matches_present = adj[list_hairpin_pairs[h_cls][0,:], 
                                        list_hairpin_pairs[h_cls][1,:]].sum()
            
            # multiply threshold by 2 since A is symmetric
            if num_matches_present > threshold*2:
                label_array[indx, h_cls] = 1
    

    # K+1th column is for "Other" class
    zeros_rows = np.where(label_array.sum(-1) == 0)
    label_array[:, K+1][zeros_rows[0]] = 1

    return label_array


def subsample_class(dataset, energy_array, labels_array, col_indx, ratio):

    N = dataset.shape[0]

    class_inds = np.arange(N)[labels_array[:,col_indx] == 1]

    non_class_inds = set(np.arange(N)) - set(class_inds)
    non_class_inds = np.array(list(non_class_inds))

    sub_inds = choice(class_inds, class_inds//ratio, replace=False)

    collate_inds = np.array(list(set(non_class_inds).union(set(sub_inds))))
    proc_inds = np.sort(collate_inds)


    subsampled_data = dataset[proc_inds]
    subsampled_energies = energy_array[proc_inds]
    subsampled_labels = labels_array[proc_inds]

    return [subsampled_data, subsampled_energies, subsampled_labels]



def generate_cls_samples(dataset, labels, cls_col_indx,  hairpin_adj):

    class_inds = np.where(labels[:,cls_col_indx] == 1)[0]

    sample_inds = choice(class_inds,3, replace=False)

    g_hair_a1 = nx.from_numpy_array(dataset[sample_inds[0]])
    g_hair_a2 = nx.from_numpy_array(dataset[sample_inds[1]])
    g_hair_a3 = nx.from_numpy_array(dataset[sample_inds[2]])
    g_a_target = nx.from_numpy_array(hairpin_adj)

    ppos = nx.kamada_kawai_layout(g_a_target)

    fig, ax = plt.subplots(2,4, figsize=(16,14))

    nx.draw(g_hair_a1, pos=ppos,node_color='k', node_size=2, ax=ax[0,0])
    nx.draw(g_hair_a2, pos=ppos,node_color='k', node_size=2, ax=ax[0,1])
    nx.draw(g_hair_a3, pos=ppos,node_color='k', node_size=2, ax=ax[0,2])
    nx.draw(g_a_target, pos=ppos, node_color='k', node_size=2, ax=ax[0,3])

    ax[1,0].imshow(dataset[sample_inds[0]])
    ax[1,1].imshow(dataset[sample_inds[1]])
    ax[1,2].imshow(dataset[sample_inds[2]])
    ax[1,3].imshow(hairpin_adj)



###########

def convertadj2db(adj):
    
    N = adj.shape[-1]
    
    adj_n = adj -  nx.adj_matrix(nx.path_graph(N)).todense()
    
    flat = adj_n.sum(-1)
    
    edge_inds = flat.nonzero()[0]
    N_edges = edge_inds.shape[0]
    open_inds = edge_inds[:N_edges//2]
    close_inds = edge_inds[N_edges//2:]
    
    db_string = list('.'*N)
    
    for i in range(N):
        if i in edge_inds:
            if i in open_inds:
                db_string[i] = '('
            if i in close_inds:
                db_string[i] = ')'

            
            
    return "".join(db_string)
            
            

