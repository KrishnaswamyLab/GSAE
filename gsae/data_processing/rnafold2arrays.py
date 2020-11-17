import numpy as np 
import re
from tqdm import tqdm
import networkx as nx
import datetime

from utils import load_data_into_arrays

import argparse


def main(infile, outfile):

    # load data from rnafold
    seq, total_fold_array, energy_array = load_data_into_arrays(infile)

    n_nodes = total_fold_array.shape[-1]

    # save into csv files
    # timestamp
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%m")

    seq_fname = "{}_sequence_{}.txt".format(outfile, date_suffix)
    adj_mat_fname = "{}_adjmat_{}.npy".format(outfile, date_suffix )
    energy_fname = "{}_energies_{}.csv".format(outfile, date_suffix)

    with open(seq_fname, "w") as seqfile:
        seqfile.write(str(seq))
    
    np.save(adj_mat_fname, total_fold_array)
    np.savetxt(energy_fname, energy_array, delimiter=",")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, 
        help='RNAfold txt file output to be converted')
    parser.add_argument('--outname', required=True,
    help='base name for the outputs')

    args = parser.parse_args()

    main(args.data, args.outname)