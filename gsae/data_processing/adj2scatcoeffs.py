
import math
import os
import glob
import random
import datetime
import numpy as np
from tqdm import tqdm

from numpy import linalg as LA
from numpy.random import choice
import scipy
from sklearn.decomposition import PCA

from gsae.scattering.scattering import transform_dataset, get_normalized_moments

import argparse


def load_dat(file):
    if file.split('.')[-1] == 'csv':
        dat = np.loadtxt(file, delimiter=',')
    elif file.split('.')[-1] == 'npy':
        dat = np.load(file)
    else:
        raise NotImplementedError
    return dat

def shift_data(data):
    """shifts data so that log transform can be applied
    """
    min_val = np.abs(data.min())

    print("shifting data by {}".format(min_val+ 0.1))

    data += min_val + 0.1

    return data



def convertadjs2scatcoeffs(adj_file, energy_file, outfname_base, n_pcs, sub_size):

    # load data
    print("loading adjacency data")
    adj_array = load_dat(adj_file)

    print("loading energies data")
    energies = load_dat(energy_file)

    print("finished loading data files")

    print("adjacency data shape: {}".format(adj_array.shape))
    print("energies data shape: {}".format(energies.shape))

    # subsample
    if sub_size > 0:
        sub_inds = np.random.choice(np.arange(adj_array.shape[0]), sub_size, replace=False)

    elif sub_size == 0:
        sub_inds = np.arange(adj_array.shape[0])

    sub_size_final = sub_inds.shape[0]

    # iterate through and transform
    print("generating scattering coefficients")
    scat_coeff_array = transform_dataset(adj_array[sub_inds])
    print("raw scattering coeffs shape: {}".format(scat_coeff_array.shape))

    # timestamp 
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%m")

    # data version 2
    # here is where statistical moments across node dimension
    print("normalizing scattering coefficients")
    norm_scat_coeffs = get_normalized_moments(scat_coeff_array).squeeze()
    print("norm scattering coeffs shape: {}".format(norm_scat_coeffs.shape))

    # data version 3
    # here is where the log transform
    print("applying log transform to scattering coefficients")
    lognorm_scat_coeffs = np.log(shift_data(norm_scat_coeffs)).squeeze()
    print("lognorm scattering coeffs shape: {}".format(lognorm_scat_coeffs.shape))

                                                            
    # PCA on scattering coefficients
    if n_pcs > 0:
        print("using {} principle components".format(n_pcs))
        
        # version 1 PCS
        base_coeff_out = PCA(n_components=int(n_pcs)).fit_transform(scat_coeff_array.reshape(sub_size_final,-1))

        # version 2 PCS
        norm_coeff_out = PCA(n_components=int(n_pcs)).fit_transform(norm_scat_coeffs)

        # version 3 PCS
        lognorm_coeff_out = PCA(n_components=int(n_pcs)).fit_transform(lognorm_scat_coeffs)
    
    
    elif n_pcs == 0:
        print("using raw scattering coefficients")

        base_coeff_out = scat_coeff_array
        norm_coeff_out = norm_scat_coeffs
        lognorm_coeff_out = lognorm_scat_coeffs


    # get shapes
    print("shape of base output: {}".format(base_coeff_out.shape))
    print("shape of norm output: {}".format(norm_coeff_out.shape))
    print("shape of lognorm output: {}".format(lognorm_coeff_out.shape))

    # create outfile names
    base_out_fname = "{}_scat_coeffs_{}_rawcoeffs.npy".format(outfname_base, date_suffix)

    norm_out_fname = "{}_normcoeffs_{}_sub{}_pca{}.npy".format(outfname_base, date_suffix,
                                                            int(sub_size_final) ,int(n_pcs))

    lognorm_out_fname = "{}_lognormcoeffs_{}_sub{}_pca{}.npy".format(outfname_base, date_suffix,
                                                            int(sub_size_final), int(n_pcs))

    energies_out_fname = "{}_energies_{}_sub{}_pca{}.npy".format(outfname_base, date_suffix,
                                                            int(sub_size_final) ,int(n_pcs))
        
    subinds_out_fname = "{}_subinds_{}_sub{}_pca{}.npy".format(outfname_base, date_suffix,
                                                            int(sub_size_final) ,int(n_pcs))

    # save scattering coefficient arrays
    np.save(base_out_fname, base_coeff_out)
    np.save(norm_out_fname, norm_coeff_out)
    np.save(lognorm_out_fname, lognorm_coeff_out)

    # save energies and subinds      
    np.save(energies_out_fname, energies[sub_inds])
    np.save(subinds_out_fname, sub_inds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, 
                        help='file (npy or csv) with adjacency matrices')
    parser.add_argument('--energies', required=True, 
                        help='file (npy or csv) with energies')
    parser.add_argument('--outname', required=True,
                        help='base name for output')
    parser.add_argument('--pcs', required=False, default=0, type=int,
                        help='how many principle components to use (if 0, then use raw scattering coefficients)')
    parser.add_argument('--subinds', required=False, default=0, type=int,
                        help='subsample size (if 0, then use all)')
    args = parser.parse_args()


    convertadjs2scatcoeffs(args.data, args.energies, args.outname, args.pcs, args.subinds)
