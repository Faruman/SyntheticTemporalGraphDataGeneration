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
from AlternativeModels.SequenceModels.wgangp import WGANGP_DSRSynthesizer
from AlternativeModels.SequenceModels.findiff import FINDIFFSynthesizer
from AlternativeModels.SequenceModels.doppelganger import DOPPELGANGERSynthesizer


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
    synthesizer = SeqSynthModel(node_embedding_dim= 6, verbose= True)
    synthesizer.fit_data_preprocessing(data= train_data, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    synthesizer.save("./synth_models/base_encoder")

train_data["timestamp"] = pd.to_datetime(train_data["timestamp"]*(10**9))

#CTGAN
if not rerun_all and os.path.exists("./synth_data/ctgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/ctgan.csv", index_col= False)
else:
    ctgan_parameters = {"embedding_dim": 64, "generator_dim": [512, 512], "discriminator_dim": [512, 512], "generator_lr": 0.00008178, "generator_decay": 0.007982, "discriminator_lr": 0.000178,
                        "discriminator_decay": 0.004898, "batch_size": 5000, "epochs": 219, "discriminator_steps": 6, "pac": 2, "verbose": True, "use_wandb": False}
    ctgan = synthesizer.fit_synth_model(data= train_data, model= CTGANSynthesizer, model_parameters= ctgan_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    ctgan.save("./synth_models/ctgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/ctgan.csv", index= False)
myBench.initialize_synth_data("CTGAN", synthetic_data, data_type= "graph-temporal")

#WGAN-GP
if not rerun_all and os.path.exists("./synth_data/wgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/wgan.csv", index_col= False)
else:
    wgangp_parameters = {"embedding_dim": 64, "generator_dim": [256, 512, 1048], "discriminator_dim": [512, 256, 128], "generator_lr": 0.0006069, "generator_decay": 0.03474, "discriminator_lr": 0.0007963,
                         "discriminator_decay": 0.04883, "batch_size": 5000, "epochs": 921, "discriminator_steps": 12, "pac": 15, "dsr_epsilon": 0.0007755, "dsr_gamma_percentile": 0.80, "verbose": True, "use_wandb": False}
    wgangp = synthesizer.fit_synth_model(data= train_data, model= WGANGP_DSRSynthesizer, model_parameters= wgangp_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    wgangp.save("./synth_models/wgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/wgan.csv", index= False)
myBench.initialize_synth_data("WGAN-GPDSR", synthetic_data, data_type= "graph-temporal")

#FinDiff
if not rerun_all and os.path.exists("./synth_data/findiff.csv"):
    synthetic_data = pd.read_csv("./synth_data/findiff.csv", index_col= False)
else:
    findiff_parameters = {"cat_embedding_dim": 8, "mlp_dim": [2048, 2048, 2048], "mlp_activation": "lrelu", "diffusion_steps": 485, "diffusion_beta_start": 0.0004387, "diffusion_beta_end": 0.005418, "mlp_lr": 0.000644,
                          "epochs": 899, "batch_size": 5000, "verbose": True, "use_wandb": False}
    findiff = synthesizer.fit_synth_model(data= train_data, model= FINDIFFSynthesizer, model_parameters= findiff_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    findiff.save("./synth_models/findiff.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/findiff.csv", index= False)
myBench.initialize_synth_data("FinDiff", synthetic_data, data_type= "graph-temporal")

#DoppelGANger
if not rerun_all and os.path.exists("./synth_data/dgan.csv"):
    synthetic_data = pd.read_csv("./synth_data/dgan.csv", index_col= False)
else:
    context_columns = data_generation_parameters["context_columns"]
    dgan_parameters = {"context_columns": context_columns, "max_sequence_len": 30, "sample_len": 10, "feature_noise_dim": 11, "attribute_noise_dim": 10, "attribute_num_layers": 3, "attribute_num_units": 139, "feature_num_layers": 5, "feature_num_units": 287,
                       "gradient_penalty_coef": 9.035, "attribute_gradient_penalty_coef": 8.21, "attribute_loss_coef": 2.048, "generator_learning_rate": 0.001833, "generator_beta1": 0.3226, "discriminator_learning_rate": 0.002267, "discriminator_beta1": 0.5658,
                       "attribute_discriminator_learning_rate": 0.000199, "attribute_discriminator_beta1": 0.3048, "discriminator_rounds": 2, "batch_size": 5000, "epochs": 575, "verbose": True, "use_wandb": False}
    dgan = synthesizer.fit_synth_model(data= train_data, model= DOPPELGANGERSynthesizer, model_parameters= dgan_parameters, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    dgan.save("./synth_models/dgan.pkl")
    synthetic_data = synthesizer.sample(len_real_data)
    synthetic_data["timestamp"] = synthetic_data["timestamp"].astype('int64') // 10**9
    synthetic_data.to_csv("./synth_data/dgan.csv", index= False)
myBench.initialize_synth_data("DoppelGANger", synthetic_data, data_type= "graph-temporal")

utility_df = myBench.evaluate_utility()
print(utility_df)

privacy_df = myBench.evaluate_privacy()
print(privacy_df)