
from argparse import ArgumentParser
import datetime
import os
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from gsae.utils import eval_metrics, load_splits
from gsae.models.gae_model import GAE
from gsae.models.gvae_model import GVAE
from gsae.models.gsae_model import GSAE



model_dict = {'gae': GAE,'gvae': GVAE}



dataset_dict = {'seq3': load_seq3,
                'seq4': load_seq4,
                'tebown': load_tebown,
                'hivtar': load_hivtar}


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--model', default='gae', type=str)
    parser.add_argument('--dataset', default='seq3', type=str)

    parser.add_argument('--input_dim', default=None, )
    parser.add_argument('--node_dim', default=50, type=int)
    parser.add_argument('--bottle_dim', default=25, type=int)
    parser.add_argument('--hidden_dim', default=400, type=int)

    parser.add_argument('--gnn_type', default='gcn', type=str)
    parser.add_argument('--pool_type', default='stat', type=str)

    parser.add_argument('--learning_rate', default=0.0001, type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--len_epoch', default=None)

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=0, type=int)
    parser.add_argument('--save_dir', default='train_logs/', type=str)


    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)


    # parse params 
    args = parser.parse_args()

    if args.model not in [ 'gvae', 'gae']:
        raise NotImplementedError

    # get data
    _, train_tup, test_tup = dataset_dict[args.dataset](gnn=true)

    # train dataset
    train_dataset = torch.utils.data.TensorDataset(*train_tup)
    test_dataset = torch.utils.data.TensorDataset(*test_tup)

    # get valid set
    train_full_size = len(train_dataset)
    train_split_size = int(train_full_size * .80)
    valid_split_size = train_full_size - train_split_size 
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_split_size, valid_split_size])

    # train loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cl_args.batch_size,
                                        shuffle=True)
    # valid loader 
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=cl_args.batch_size,
                                        shuffle=False)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir =  args.save_dir + '{}_logs_run_{}_{}/'.format(args.model,args.dataset,date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='min'
            )

    # init module
    args.input_dim = train_tup[0].shape[-1]
    args.len_epoch = len(train_loader)

    model = model_dict[args.model](hparams=args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.n_epochs,
                                             gpus=args.n_gpus,
                                            callbacks=[early_stop_callback])
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloaders=valid_loader,
                )

    model = model.cpu()
    model.dev_type = 'cpu'

    with torch.no_grad():
        train_embed = model.embed(train_tup[0])[0]
        test_embed =  model.embed(test_tup[0])[0]

    # save data
    print('saving embeddings')
    np.save(save_dir + "train_embedding.npy" , train_embed.cpu().detach().numpy() )
    np.save(save_dir + "test_embedding.npy" , test_embed.cpu().detach().numpy() )


    # EVALUATION ON TEST SET 
    # energy pred mse

    print("getting test set predictions")
    with torch.no_grad():
        x_recon_test = model(test_tup[0])[0]
        y_pred_test = model.predict_from_data(test_tup[0])


    pred_test_val = nn.MSELoss()(y_pred_test.flatten(), test_tup[-1].flatten())




    print("gathering eval subsets")
    eval_tup_list = [eval_metrics.compute_subsample([test_embed, test_tup[-1]], 10000)[0] for i in range(8)]
    # trainer.test()
    print("getting smoothness vals")
    embed_eval_array= np.expand_dims(np.array([x[0].numpy() for x in eval_tup_list]),0)
    energy_eval_array= np.array([x[1].numpy() for x in eval_tup_list])

    print('embed_eval_array shape: {}'.format(embed_eval_array.shape))
    print('energy_eval_array shape: {}'.format(energy_eval_array.shape))
    
    energy_smoothness = eval_metrics.eval_over_replicates(embed_eval_array,
                                                            energy_eval_array,
                                                eval_metrics.get_smoothnes_kNN,
                                                [5, 10])[0]

    energy_smoothness = eval_metrics.format_metric(energy_smoothness)
