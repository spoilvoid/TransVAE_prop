[![DOI](https://zenodo.org/badge/287491872.svg)](https://zenodo.org/badge/latestdoi/287491872)
# Giving Attention to Generative VAE Models for _De Novo_ Molecular Design
![Attention Heads](https://raw.githubusercontent.com/oriondollar/TransVAE/master/imgs/attn_heads.png)
This repo is copied from [oriondollar/TransVAE](https://github.com/oriondollar/TranSVAE) and contains the codebase for the attention-based implementations of VAE models for molecular design as described in [this paper](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc01050f). The code is organized by folders that correspond to the following sections:

- **transvae**: code required to run models including model class definitions, data preparation, optimizers, etc.
- **scripts**: scripts for training models, generating samples and performing calculations
- **notebooks**: jupyter notebook tutorials and example calculations
- **checkpoints**: pre-trained model files
- **data**: token vocabularies and weights for ZINC and PubChem datasets (***note - full train and test sets for both ZINC and PubChem)

## Installation

The code can be installed with pip using the following command `pip install transvae`. But it's inconvenient to modify source code and customize data. Thus you can follow below commands 

```
git clone https://github.com/spoilvoid/TransVAE_prop.git
conda create -n TransVAE python=3.8
conda activate TransVAE
cd TransVAE
pip install -r requirements.txt
```

If your CUDA capability ${config_name} is not compatible with the current PyTorch installation, you should change torch's version in `requirements.txt`or just excludes it and pip install specific PyTorch version.

 [RDKit](https://www.rdkit.org/docs/Install.html) is required for property calculation and sample. You can install it by 

  `pip install rdkit`

 [tensor2tensor](https://github.com/tensorflow/tensor2tensor) is required for certain visualizations. You can install it by

 `pip install tensor2tensor`

 Neither of these packages are necessary for training or generating molecules so if you would prefer not to install them then you can simply remove their imports from the source code.

 Finally, you should run below command to configure the Project

 `python setup.py develop`

 If some large files isn't downloaded, please check your git or mannually download it by yourself.
 
## Training for Property Prediction

![Model Types](https://raw.githubusercontent.com/oriondollar/TransVAE/master/imgs/model_types.png)

There are three model types - RNN (a), RNNAttn (b) and Transformer (c). *The model paper proposed is Transformer (c)*. 

To train an VAE with Property Prediction tasks, an additional set of linear layers (MLP) should be appended to the latent memory to embed a property within the bottleneck using the `property_predictor` tag. And you must supply an additional set of train and test files with properties indexed at the same position as the molecules in the train and test sets (every .txt file should hold only one column for target data).

As for what kind of property, you can choose by yourself and complement it with the help of `rdkit`. You should put it in `data` folder. 1st row should be head (your property name) and others follow property value per row which corresponds to SMILES file.

A command to train a model with this functionality might look like If you've downloaded the $ZINC$ or $PubChem$ training sets from the [original project](https://github.com/oriondollar/TranSVAE)'s drive link, a command to train a model with this functionality might look like

 `python scripts/train.py --model transvae --property_predictor --data_source zinc --train_props_path ${train_property_filename}.txt --test_props_path ${test_property_filename}.txt --save_name ${my_props_model}`

In above command, `--data_source` can choose $zinc$ or $pubchem$. In `./data` folder, we provide a dataset of $ZINC250k subset$. Thus if you want to run default data directly or use custom data, please refer `train_prop.sh` or specify a custom train and test set like so

 `python scripts/train.py --model transvae --property_predictor --data_source custom --train_mols_path ${mols_train_filename}.txt --test_mols_path ${mols_test_filename}.txt --vocab_path ${my_vocab}.pkl --char_weights_path ${my_char_weights}.npy --train_props_path ${train_property_filename}.txt --test_props_path ${test_property_filename}.txt --save_name ${my_props_model}`

The DEFAULT data directory is `./data`. And the vocabulary must be a pickle file that stores a dictionary that maps token -> tokenid and it must begin with the `<start>` or `<bos>` token. All modifiable hyperparameters can be viewed with `python scripts/train.py --help`. The above command need a GPU to train the model or just run in cpu (but quite slower).

If you have trained for several epochs and want to resume from ckpt to continue training, you can achieve it by specifying the args `--checkpoint ${ckpt_path}`. Because .ckpt file has recorded the epoch information, your start epoch will be `ckpt['epoch']`.

## Evaluation
 ## Property Prediction
 There is no direct file to run evaluation of Property Prediction. But it's easy to follow `./scripts/sample.py` to write a `predict.py`, just use

 ```
 src = Variable(mols_data).long()
 src_mask = (src != self.pad_idx).unsqueeze(-2)
 _, mu, _, _ = vae.model.encode(src, src_mask)
 pred_props = vae.model.predict_property(mu)
 ```

 You may need to implement it by yourself or **we offer one source code**.

 We offer `predict.py` in **scripts** folder. An example test command might look like:

`python scripts/predict.py --model transvae --model_ckpt checkpoints/${prop}.ckpt --mols ${mols_test_filename}.txt --props ${test_property_filename}.txt --save_path ${your_save_path}.txt`

 ## Sampling

 There are three sampling modes to choose from - random, high entropy or k-random high entropy. If you choose to use one of the high entropy categories, you must also supply a set of SMILES (typically the training set) to use to calculate the entropy of your model prior to sampling. An example command might look like:

 `python scripts/sample.py --model transvae --model_ckpt checkpoints/trans4x-256_zinc.ckpt --mols data/zinc_train.txt --sample_mode high_entropy`

 ## Calculating Attention

 Attention can be calculated using the `attention.py` script. Due to the large number of attention heads and layers within the transvae model you should be careful about calculating attention for too many samples as it will generate a large amount of data. An example command for calculating attention might look like

 `python scripts/attention.py --model rnnattn --model_ckpt checkpoints/rnnattn-256_pubchem.ckpt --mols data/pubchem_train_(n=500).txt --save_path attn_wts/rnnattn_wts.npy`

 ## Analysis

 Examples of model analysis functions and how to use them are shown in `notebooks/visualizing_attention.ipynb` and `notebooks/evaluating_models.ipynb`. Additionally, there are a few helper functions in `transvae/analysis.py` that allow you to plot training performance curves and other useful performance metrics.

 ![Training Curve](https://raw.githubusercontent.com/oriondollar/TransVAE/master/imgs/training_curve.png)
