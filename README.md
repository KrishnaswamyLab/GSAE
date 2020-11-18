
<div align="center">    
 
# Uncovering the Folding Landscape of RNA Secondary Structure with Deep Graph Embeddings
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2006.06885)

[![Conference](http://img.shields.io/badge/ICLR-GRL+-2020-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
 -->

[![Paper](https://img.shields.io/badge/arxiv-2006.06885-B31B1B.svg)](https://arxiv.org/abs/2006.06885)




<!--  
Conference   
-->   
</div>
 
## Visual Description   
![visual overview](./assets/overview.png)

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ec134/GSAE   

# install project   
cd GSAE
pip install -e .   
pip install -r requirements.txt
 ```   
 
 
# Workflow

## loading data from RNAfold
---


From rnafold, we get a file like the following

    rnafold_output.txt

which inside looks like

    GGCGUUUUCGCCUUUUGGCGAUUUUUAUCGCC -14.20  10.00
    (.((...(((((....))))).......)).)  -5.50
    (.(((..(((((....))))).....).)).)  -4.20
    ((.....(((((....))))).........))  -5.90
    ((.((..(((((....))))).....))..))  -5.60
    ((.(...(((((....))))).......).))  -6.40
    ((.(.(.(((((....))))).....).).))  -4.20
    ((.((..(((((....))))).....).).))  -5.10
    (((....(((((....))))).(...)..)))  -4.30
    (((....(((((....)))))(....)..)))  -5.30

We can use `rnafold2arrays.py` in `gsae/data_processing` to convert this text file to

- a csv file containing adjacency matrices for each fold (`adjmats_<datestamp>.csv`)
- a csv file containing the energy scalar for each structure (`energies_<datestamp>.csv`)
- a text file with the rna sequence (`sequence_<datestamp>.txt`)

`rnafold2arrays.py` usage:

    usage: rnafold2arrays.py [-h] --data DATA --outname OUTNAME

    optional arguments:
    -h, --help         show this help message and exit
    --data DATA        RNAfold txt file output to be converted
    --outname OUTNAME  base name for the outputs

sample usage:

    > python rnafold2arrays.py --data seq4_rnafold_out.txt --outname seq4
    > ls
    seq4_adjmat_2020-03-04-03.csv
    seq4_energies_2020_03-04-03.csv
    seq4_sequence_2020-03-04-03.txt
    rnafold2arrays.py
    seq3_rnafold_out.py


## Converting adjacency data to scattering coefficients
---


Once we have the adjacency matrices of the structures we're interested in, we can convert them using scattering transforms to a new, more informative representation

Here we will use diracs centered at each node (i.e. the identity matrix) as our graph signals.

To convert them, we will use `adj2scatcoeffs.py`

`adj2scatcoeffs.py` usage:

    usage: adj2scatcoeffs.py [-h] --data DATA --outname OUTNAME --graph_size
                            GRAPH_SIZE [--pcs PCS]

    optional arguments:
    -h, --help            show this help message and exit
    --data DATA           csv with adjacency matrices
    --outname OUTNAME     base name for output
    --graph_size GRAPH_SIZE
                            number of nodes in graphs (assume equal size)
    --pcs PCS             how many principle components to use (if 0, then use raw scattering coefficients)

sample usage:

    > python adj2scatcoeffs.py --data seq4_adjmat_2020-03-04-03.csv --outname seq4 --graph_size 32 --pcs 100

    > ls
    seq4_scat_coeff_2020-03-04-03_pca_n100.csv
    seq4_adjmat_2020-03-04-03.csv
    adj2scatcoeffs.py


## Data

Data for the 4 sequences used in the paper are located in data/


    └── raw_data
        ├── hiv_tar
        │   ├── hiv_tar_sequence.txt
        │   ├── hivtar_100k_subp_n_052020.txt
        ├── hob_seq3
        │   ├── seq3_100k_subp_n_052020.txt
        │   └── seq3_sequence.txt
        ├── hob_seq4
        │   ├── seq4_100k_subp_n_052020.txt
        │   └── seq4_sequence.txt
        └── tebown
            ├── teb_100k_subp_n_052020.txt
            └── tebown_sequence.txt


## IMPORTANT: Data loading for models

In order to ensure that the training scripts in the model files function correctly, the global variables at the top of `load_splits.py`  must be assigned to whereever you save the outputs of `adj2scatcoeffs.py`. 


## Training the models

    python gsae/models/gsae_model.py 

  
arguments:

    usage: gsae_model.py [--input_dim INPUT_DIM] [--dataset DATASET] [--bottle_dim BOTTLE_DIM] [--hidden_dim HIDDEN_DIM] [--learning_rate LEARNING_RATE] [--alpha ALPHA] [--beta BETA] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--n_gpus N_GPUS] [--save_dir SAVE_DIR]


### Citation   
```
@article{castro2020uncovering,
  title={Uncovering the Folding Landscape of RNA Secondary Structure with Deep Graph Embeddings},
  author={Castro, Egbert and Benz, Andrew and Tong, Alexander and Wolf, Guy and Krishnaswamy, Smita},
  journal={arXiv preprint arXiv:2006.06885},
  year={2020}
}
```   
