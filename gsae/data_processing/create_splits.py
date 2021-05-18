
import numpy as np

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from gsae.data_processing.utils import load_dat


def split_data(adjs, coeffs, energies):

    split_data = [adjs, coeffs,  energies ]
    tr_adjs, te_adjs, tr_coeffs, te_coeffs, tr_energies, te_energies = train_test_split(*split_data,
                                                                    train_size=0.70,
                                                                    random_state=42)    


    return [tr_adjs, tr_coeffs, tr_energies], [te_adjs, te_coeffs, te_energies]




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--adjs', required=True, 
                        help='file with adjacency matrices')
    parser.add_argument('--coeffs', required=True, 
                        help='file with scattering coeffs')
    parser.add_argument('--energies', required=True, 
                        help='file with energy values')
    parser.add_argument('--outname', required=True,
                        help='base name for output')
    args = parser.parse_args()

    print('loading files')
    adj_array = load_dat(args.adjs)
    coeff_array = load_dat(args.coeffs)
    energies_array = load_dat(args.energies)
    print('finished loading files')

    train_data, test_data = split_data(adj_array, coeff_array, energies_array)

    print('saving splits')
    for x, n in zip(train_data, ['adjs', 'coeffs', 'energies']):
        np.save(f'{args.outname}_train_{n}.npy', x)

    for x, n in zip(test_data, ['adjs', 'coeffs', 'energies']):
        np.save(f'{args.outname}_test_{n}.npy', x)

    print('finished')
