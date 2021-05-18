
import datetime
import numpy as np
from sklearn.decomposition import PCA

from gsae.scattering.scattering import transform_dataset, get_normalized_moments
from gsae.data_processing.utils import load_dat
import argparse



def shift_data(data):
    """shifts data so that log transform can be applied
    """
    min_val = np.abs(data.min())

    print("shifting data by {}".format(min_val+ 0.1))

    data += min_val + 0.1

    return data


def convertadjs2scatcoeffs(adj_file, outfname_base, n_pcs):

    # load data
    print("loading adjacency data")
    adj_array = load_dat(adj_file)

    print("adjacency data shape: {}".format(adj_array.shape))


    # iterate through and transform
    print("generating scattering coefficients")
    scat_coeff_array = transform_dataset(adj_array)
    print("raw scattering coeffs shape: {}".format(scat_coeff_array.shape))

    # timestamp 
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%m")

    # data version 2
    # here is where statistical moments across node dimension
    print("normalizing scattering coefficients")
    norm_scat_coeffs = get_normalized_moments(scat_coeff_array).squeeze()
    print("norm scattering coeffs shape: {}".format(norm_scat_coeffs.shape))

                                                            
    # PCA on scattering coefficients
    if n_pcs > 0:
        print("using {} principle components".format(n_pcs))
        
        # version 1 PCS
        base_coeff_out = PCA(n_components=int(n_pcs)).fit_transform(scat_coeff_array.reshape(sub_size_final,-1))

        # version 2 PCS
        norm_coeff_out = PCA(n_components=int(n_pcs)).fit_transform(norm_scat_coeffs)


    
    elif n_pcs == 0:
        print("using raw scattering coefficients")

        base_coeff_out = scat_coeff_array
        norm_coeff_out = norm_scat_coeffs


    # get shapes
    print("shape of base output: {}".format(base_coeff_out.shape))
    print("shape of norm output: {}".format(norm_coeff_out.shape))
    # print("shape of lognorm output: {}".format(lognorm_coeff_out.shape))

    # create outfile names
    # base_out_fname = "{}_scat_coeffs_{}_rawcoeffs.npy".format(outfname_base, date_suffix)

    norm_out_fname = "{}_normcoeffs_{}_pca{}.npy".format(outfname_base, date_suffix,
                                                            int(n_pcs))
        


    # save scattering coefficient arrays
    # np.save(base_out_fname, base_coeff_out)
    np.save(norm_out_fname, norm_coeff_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, 
                        help='file (npy or csv) with adjacency matrices')
    parser.add_argument('--outname', required=True,
                        help='base name for output')
    parser.add_argument('--pcs', required=False, default=0, type=int,
                        help='how many principle components to use (if 0, then use raw scattering coefficients)')
    args = parser.parse_args()


    convertadjs2scatcoeffs(args.data, args.outname, args.pcs)
