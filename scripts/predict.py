import os
import argparse
import pickle
import pkg_resources
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from transvae.tvae_util import calc_entropy
from transvae.data import vae_data_gen

def predict(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    elif args.model == 'rnn':
        vae = RNN(load_fn=ckpt_fn)
    
    if vae.model.property_predictor is None:
        raise ValueError('Model does not have a property predictor. Please use a model with a property predictor.')
    
    test_mols = pd.read_csv(args.mols).to_numpy()
    test_props = pd.read_csv(args.props).to_numpy()

    test_data = vae_data_gen(test_mols, test_props, char_dict=vae.params['CHAR_DICT'])
    test_iter = torch.utils.data.DataLoader(test_data,
                                               batch_size=vae.params['BATCH_SIZE'],
                                               shuffle=False, num_workers=0,
                                               pin_memory=False, drop_last=True)
    chunk_size = vae.params['BATCH_SIZE'] // vae.params['BATCH_CHUNKS']
    torch.backends.cudnn.benchmark = True

    pred_prop_list, real_prop_list = [], []
    vae.model.eval()
    for j, data in enumerate(test_iter):
        for i in range(vae.params['BATCH_CHUNKS']):
            batch_data = data[i*chunk_size:(i+1)*chunk_size,:]
            mols_data = batch_data[:,:-1]
            props_data = batch_data[:,-1]
            if vae.use_gpu:
                mols_data = mols_data.cuda()
                props_data = props_data.cuda()

            src = Variable(mols_data).long()
            true_prop = Variable(props_data)
            src_mask = (src != vae.pad_idx).unsqueeze(-2)
            _, mu, _, _ = vae.model.encode(src, src_mask)
            pred_props = vae.model.predict_property(mu)
            pred_prop_list.extend(pred_props.reshape(-1).cpu().detach().numpy())
            real_prop_list.extend(true_prop.reshape(-1).cpu().detach().numpy())
    pred_prop_list = np.array(pred_prop_list)
    real_prop_list = np.array(real_prop_list)
    mse_loss = np.mean((pred_prop_list - real_prop_list) ** 2)
    print(f'MSE Loss: {mse_loss}')
    with open(args.save_path, 'w') as f:
        f.write('pred_prop' + '\n')
        for prop in pred_prop_list:
            f.write(str(prop) + '\n')


def predict_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', required=True, type=str)
    parser.add_argument('--props', required=True, type=str)
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser


if __name__ == '__main__':
    parser = predict_parser()
    args = parser.parse_args()
    predict(args)
