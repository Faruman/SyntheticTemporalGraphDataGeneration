import os.path
import copy
import datetime

import numpy as np
import pandas as pd

import timeit

from sklearn.model_selection import train_test_split

from TGSynthBench import TGSynthBench
from AlternativeModels.SequenceModels.base import SeqSynthModel

from AlternativeModels.SequenceModels.ctgan import CTGANSynthesizer
from AlternativeModels.SequenceModels.tvae import TVAESynthesizer
from AlternativeModels.SequenceModels.wgangp import WGANGP_DSRSynthesizer
from AlternativeModels.SequenceModels.findiff import FINDIFFSynthesizer
from AlternativeModels.SequenceModels.doppelganger import DOPPELGANGERSynthesizer
from AlternativeModels.SequenceModels.timegan import TIMEGANSynthesizer


rerun_all = False

real_data = pd.read_csv("./data/UBP.csv", index_col= False)
#real_data = real_data.sample(frac= 0.2, random_state= 42)

print("Real data shape: ", real_data.shape)
len_real_data = len(real_data)

data_generation_parameters = {"sender_column": "source_id", "receiver_column": "target_id", "sender_attributes": ['PCACCT1', 'PCACCT2', 'PCACCT3', 'PCACCT4', 'PCACCT5', 'PCACCT6', 'PCACCT7', 'PCACCT8', 'PCACCT9', 'PCACCT10', 'PCACCT11', 'PCACCT12', 'PCACCT13', 'PCACCT14', 'PCACCT15', 'PCACCT16', 'PCACCT17', 'PCACCT18', 'PCACCT19', 'PCACCT20', 'PCACCT21', 'PCACCT22', 'PCACCT23'],
                              "min_seq_len": 3, "max_seq_len": 20,
                              "graph_emb_num_neighbours": 150, "graph_emb_num_neighbours_step": 3, "graph_emb_learningrate": 0.01, "graph_emb_epochs": 30,
                              "seq_emb_learning_rate": 0.01, "seq_emb_epochs": 30}

myBench = TGSynthBench(data_type= "graph-temporal", data_path="./bench/data", model_path = "./bench/models/", result_path = "./bench/results/", device= "cuda:0", seed= 42)
myBench.initialize_data(real_data=real_data, train_val_test_split = (0.75, 0.1, 0.15), split_type="time", time_column="timestamp", label_column="fraud", normalize_data=True, data_generation_parameters=data_generation_parameters)


train_data = myBench.get_train_data(data_type= "tabular").get_dataframe(return_time= True, return_edge_index= True)
if not rerun_all and os.path.exists("./synth_models/base_encoder/meta.pkl"):
    synthesizer = SeqSynthModel()
    synthesizer.load("./synth_models/base_encoder")
else:
    synthesizer = SeqSynthModel(node_embedding_dim= 6, device="cuda:1", verbose= True)
    synthesizer.fit_data_preprocessing(data= train_data, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    synthesizer.save("./synth_models/base_encoder")

train_data["timestamp"] = pd.to_datetime(train_data["timestamp"]*(10**9))

#CTGAN
print("Run CTGAN")
if not rerun_all and os.path.exists("./synth_data/ctgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/ctgan.csv", index_col= False)
else:
    ctgan_parameters = {'pac': 1, 'epochs': 301, 'batch_size': 3200, 'generator_lr': 0.0007073853917769696, 'embedding_dim': 32, 'generator_dim': [512, 512], 'generator_decay': 0.048019425560661466, 'discriminator_lr': 0.000971514627314732, 'discriminator_dim': [512, 512], 'discriminator_decay': 0.024051723376974643, 'discriminator_steps': 12, "verbose": True, "use_wandb": False}
    ctgan = synthesizer.fit_synth_model(data= train_data, model= CTGANSynthesizer, model_parameters= ctgan_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    ctgan.save("./synth_models/ctgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/ctgan.csv", index= False)
myBench.initialize_synth_data("CTGAN", synthetic_data, data_type= "graph-temporal")

#TVAE
print("Run TVAE")
if not rerun_all and os.path.exists("./synth_data/tvae.csv"):
    synthetic_data = pd.read_csv("./synth_data/tvae.csv", index_col= False)
else:
    tvae_parameters = {'epochs': 732, 'l2scale': 0.000345222922180821, 'batch_size': 5000, 'loss_factor': 5, 'compress_dims': [128, 128], 'embedding_dim': 256, 'learning_rate': 0.00034051342302598357, 'decompress_dims': [128, 128], "verbose": True, "use_wandb": False}
    tvae = synthesizer.fit_synth_model(data= train_data, model= TVAESynthesizer, model_parameters= tvae_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    tvae.save("./synth_models/tvae.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/tvae.csv", index= False)
myBench.initialize_synth_data("TVAE", synthetic_data, data_type= "graph-temporal")

#WGAN-GP with DRS
print("Run WGAN-GP with DRS")
if not rerun_all and os.path.exists("./synth_data/wgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/wgan.csv", index_col= False)
else:
    wgangp_parameters = {'pac': 4, 'epochs': 448, 'batch_size': 3200, 'dsr_epsilon': 0.0009414671790733584, 'generator_lr': 0.0004042104608623104, 'embedding_dim': 32, 'generator_dim': [128, 256, 512], 'generator_decay': 0.04892965958149615, 'discriminator_lr': 0.0008379006471216435, 'discriminator_dim': [512, 256, 128], 'discriminator_decay': 0.04246150001066506, 'discriminator_steps': 14, 'dsr_gamma_percentile': 0.7237222641936131, "verbose": True, "use_wandb": False}
    wgangp = synthesizer.fit_synth_model(data= train_data, model= WGANGP_DSRSynthesizer, model_parameters= wgangp_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    wgangp.save("./synth_models/wgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/wgan.csv", index= False)
myBench.initialize_synth_data("WGAN-GPDSR", synthetic_data, data_type= "graph-temporal")

#FinDiff
print("Run FinDiff")
if not rerun_all and os.path.exists("./synth_data/findiff.csv"):
    synthetic_data = pd.read_csv("./synth_data/findiff.csv", index_col= False)
else:
    findiff_parameters = {'epochs': 313, 'mlp_lr': 0.0004176927709974139, 'mlp_dim': [1024, 1024, 1024, 1024], 'batch_size': 3200, 'mlp_activation': 'tanh', 'diffusion_steps': 683, 'cat_embedding_dim': 2, 'diffusion_beta_end': 0.05658133918732594, 'diffusion_beta_start': 0.0007677975084912767, "verbose": True, "use_wandb": False}
    findiff = synthesizer.fit_synth_model(data= train_data, model= FINDIFFSynthesizer, model_parameters= findiff_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    findiff.save("./synth_models/findiff.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/findiff.csv", index= False)
myBench.initialize_synth_data("FinDiff", synthetic_data, data_type= "graph-temporal")

train_data["timestamp"] = train_data["timestamp"].astype('int64') // 10 ** 9

#DoppelGANger
print("Run DoppelGANger")
if not rerun_all and os.path.exists("./synth_data/dgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/dgan.csv", index_col= False)
else:
    context_columns = data_generation_parameters["sender_attributes"]
    dgan_parameters = {'epochs': 153, 'batch_size': 3200, 'sample_len': 3, 'generator_beta1': 0.8685460722608853, 'feature_noise_dim': 6, 'feature_num_units': 280, 'feature_num_layers': 3, 'attribute_loss_coef': 2.052392825271194, 'attribute_noise_dim': 9, 'attribute_num_units': 462, 'discriminator_beta1': 0.8929314308298053, 'attribute_num_layers': 2, 'discriminator_rounds': 2, 'gradient_penalty_coef': 14.750293307389734, 'generator_learning_rate': 0.00093366554349798, 'discriminator_learning_rate': 0.00033186069460119553, 'attribute_discriminator_beta1': 0.5284908048806307, 'attribute_gradient_penalty_coef': 10.888090437287175, 'attribute_discriminator_learning_rate': 0.00497619882802683, "context_columns": context_columns, "verbose": True, "use_wandb": False}
    dgan = synthesizer.fit_synth_model(data= train_data, model= DOPPELGANGERSynthesizer, model_parameters= dgan_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]], model_type= "sequential")
    dgan.save("./synth_models/dgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data.to_csv("./synth_data/dgan.csv", index= False)
myBench.initialize_synth_data("DoppelGANger", synthetic_data, data_type= "graph-temporal")

#TimeGAN
print("Run TimeGAN")
if not rerun_all and os.path.exists("./synth_data/timegan.csv"):
    synthetic_data = pd.read_csv("./synth_data/timegan.csv", index_col= False)
else:
    context_columns = data_generation_parameters["sender_attributes"]
    timegan_parameters = {'beta1': 0.8520381900946721, 'gamma': 9.025765337196828, 'epochs': 303, 'batch_size': 256, 'hidden_dim': 64, 'num_layers': 2, 'learning_rate': 0.004231397764479854, 'generator_steps': 1, 'metric_iteration': 15, 'encoder_loss_weight_0': 13.970338919180442, 'encoder_loss_weight_s': 0.8291164644636484, 'generator_loss_weight': 145, "context_columns": context_columns, "verbose": True, "use_wandb": False}
    timegan = synthesizer.fit_synth_model(data= train_data, model= TIMEGANSynthesizer, model_parameters= timegan_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]], model_type= "sequential")
    timegan.save("./synth_models/timegan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data.to_csv("./synth_data/timegan.csv", index= False)
myBench.initialize_synth_data("TimeGAN", synthetic_data, data_type= "graph-temporal")

utility_df = myBench.evaluate_utility()
print(utility_df)

privacy_df = myBench.evaluate_privacy()
print(privacy_df)