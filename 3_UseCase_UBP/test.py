import numpy as np
import pandas as pd

import timeit

from sklearn.model_selection import train_test_split

from TGSynthBench import TGSynthBench

class testSynthesizer():
    def __init__(self, data):
        self.data = data
        self.data["source_id"] = self.data["source_id"] + self.data["source_id"].max()
        self.data["target_id"] = self.data["target_id"] + self.data["target_id"].max()
    def sample(self, num_rows):
        if num_rows > len(self.data):
            samples = []
            sample_lens = []
            while sum(sample_lens) < num_rows:
                if num_rows-sum(sample_lens) >= len(self.data):
                    samples.append(self.data.sample(len(self.data), random_state=24))
                    sample_lens.append(len(self.data))
                else:
                    samples.append(self.data.sample(num_rows - sum(sample_lens), random_state=24))
                    sample_lens.append(num_rows - sum(sample_lens))
            return pd.concat(samples)
        else:
            return self.data.sample(num_rows)
    def save(self, path):
        pass


real_data = pd.read_csv("./data/UBP.csv", index_col= False)

real_data, test_real_data = train_test_split(real_data, test_size=0.5, random_state=42)

print("Real data shape: ", real_data.shape)
print("Test data shape: ", test_real_data.shape)

data_generation_parameters = {"sender_column": "source_id", "receiver_column": "target_id", "sender_attributes": ['PCACCT1', 'PCACCT2', 'PCACCT3', 'PCACCT4', 'PCACCT5', 'PCACCT6', 'PCACCT7', 'PCACCT8', 'PCACCT9', 'PCACCT10', 'PCACCT11', 'PCACCT12', 'PCACCT13', 'PCACCT14', 'PCACCT15', 'PCACCT16', 'PCACCT17', 'PCACCT18', 'PCACCT19', 'PCACCT20', 'PCACCT21', 'PCACCT22', 'PCACCT23'],
                              "min_seq_len": 3, "max_seq_len": 20,
                              "graph_emb_num_neighbours": 150, "graph_emb_num_neighbours_step": 3, "graph_emb_learningrate": 0.01, "graph_emb_epochs": 30,
                              "seq_emb_learning_rate": 0.01, "seq_emb_epochs": 30}

myBench = TGSynthBench(data_type= "graph-temporal", data_path="./bench/data", model_path = "./bench/models/", result_path = "./bench/results/", device= "cuda:0", seed= 42)
myBench.initialize_data(real_data=real_data, train_val_test_split = (0.75, 0.1, 0.15), split_type="time", time_column="timestamp", label_column="fraud", normalize_data=True, data_generation_parameters=data_generation_parameters)


real_train_data = myBench.get_train_data(data_type= "tabular")
myTestSynthesizer = testSynthesizer(data= test_real_data)
synth_data = myTestSynthesizer.sample(num_rows= myBench.real_data_size[0])
myBench.initialize_synth_data("TestSynthesizer", synth_data, data_type= "graph-temporal")


utility_df = myBench.evaluate_utility()
print(utility_df)

privacy_df = myBench.evaluate_privacy()
print(privacy_df)