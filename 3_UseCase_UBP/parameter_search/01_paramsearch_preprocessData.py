import os
import pandas as pd

from TGSynthBench import TGSynthBench
from AlternativeModels.SequenceModels.base import SeqSynthModel


real_data = pd.read_csv("../data/UBP.csv", index_col= False)
#real_data = real_data.sample(frac= 0.2, random_state= 42)

print("Real data shape: ", real_data.shape)
len_real_data = len(real_data)

data_generation_parameters = {"sender_column": "source_id", "receiver_column": "target_id", "sender_attributes": ['PCACCT1', 'PCACCT2', 'PCACCT3', 'PCACCT4', 'PCACCT5', 'PCACCT6', 'PCACCT7', 'PCACCT8', 'PCACCT9', 'PCACCT10', 'PCACCT11', 'PCACCT12', 'PCACCT13', 'PCACCT14', 'PCACCT15', 'PCACCT16', 'PCACCT17', 'PCACCT18', 'PCACCT19', 'PCACCT20', 'PCACCT21', 'PCACCT22', 'PCACCT23'],
                              "min_seq_len": 3, "max_seq_len": 20,
                              "graph_emb_num_neighbours": 150, "graph_emb_num_neighbours_step": 3, "graph_emb_learningrate": 0.01, "graph_emb_epochs": 30,
                              "seq_emb_learning_rate": 0.01, "seq_emb_epochs": 30}

myBench = TGSynthBench(data_type= "graph-temporal", data_path="../bench/data", model_path = "../bench/models/", result_path = "../bench/results/", device= "cuda:0", seed= 42)
myBench.initialize_data(real_data=real_data, train_val_test_split = (0.75, 0.1, 0.15), split_type="time", time_column="timestamp", label_column="fraud", normalize_data=True, data_generation_parameters=data_generation_parameters)


train_data = myBench.get_train_data(data_type= "tabular").get_dataframe(return_time= True, return_edge_index= True)
if os.path.exists("../synth_models/base_encoder/meta.pkl"):
    synthesizer = SeqSynthModel()
    synthesizer.load("../synth_models/base_encoder")
else:
    synthesizer = SeqSynthModel(node_embedding_dim= 6, verbose= True)
    synthesizer.fit_data_preprocessing(data= train_data, edge_index_cols= [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]])
    synthesizer.save("../synth_models/base_encoder")

train_data["timestamp"] = pd.to_datetime(train_data["timestamp"]*(10**9))

for col in [data_generation_parameters["sender_column"], data_generation_parameters["receiver_column"]]:
    train_data[col] = train_data[col].map(synthesizer.node_relabel_dict)
    train_data = pd.concat((train_data, pd.DataFrame(synthesizer.embeddings[train_data[col].values], columns=[f"{col}_WYS_{i}" for i in range(synthesizer.node_embedding_dim)])), axis=1)
    train_data = train_data.drop(columns=[col])

train_data.to_csv("./temp/train_data.csv", index=False)