import pandas as pd

from sklearn.model_selection import train_test_split

# Importing the benchmarking framework for synthetic data generation and evaluation
from TGSynthBench import TGSynthBench


# A mock synthesizer class used for testing (doesn't train a model, just resamples data)
class testSynthesizer():
    def __init__(self, data):
        self.data = data
        self.data["Sender"] = self.data["Sender"] + self.data["Sender"].max()
        self.data["Receiver"] = self.data["Receiver"] + self.data["Receiver"].max()
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

# Load the IBM-AML transactional dataset
real_data = pd.read_csv("./data/IBM-AML_small.csv", index_col= False)

# Split the real data into a train and test set (50/50)
real_data, test_real_data = train_test_split(real_data, test_size=0.5, random_state=42)

print("Real data shape: ", real_data.shape)
print("Test data shape: ", test_real_data.shape)

# Define parameters used for data preparation and embeddings
data_generation_parameters = {"sender_column": "Sender", "receiver_column": "Receiver",
                              "sender_attributes": ['Sender_PCA0', 'Sender_PCA1', 'Sender_PCA2'],
                              "sender_attributes_prefix": "Sender_",
                              "receiver_attributes": ['Receiver_PCA0', 'Receiver_PCA1', 'Receiver_PCA2'],
                              "receiver_attributes_prefix": "Receiver_",
                              "min_seq_len": 3, "max_seq_len": 20,
                              "graph_emb_num_neighbours": 150, "graph_emb_num_neighbours_step": 3, "graph_emb_learningrate": 0.01, "graph_emb_epochs": 30,
                              "seq_emb_learning_rate": 0.01, "seq_emb_epochs": 30}

# Initialize the benchmarking suite with paths, device, and seed
myBench = TGSynthBench(data_type= "graph-temporal", data_path="./bench/data", model_path = "./bench/models/", result_path = "./bench/results/", device= "cuda:1", seed= 42)
myBench.initialize_data(real_data=real_data, train_val_test_split = (0.75, 0.1, 0.15), split_type="time", time_column="Timestamp", label_column="Target", normalize_data=True, data_generation_parameters=data_generation_parameters)

# Initialize the Mock synthesizer
real_train_data = myBench.get_train_data(data_type= "tabular")
myTestSynthesizer = testSynthesizer(data= test_real_data)
synth_data = myTestSynthesizer.sample(num_rows= myBench.real_data_size[0])
myBench.initialize_synth_data("TestSynthesizer", synth_data, data_type= "graph-temporal")

# Run evaluation
result = myBench.run_complete_evaluation(privacy_run_mia= True, utility_run_sds= True, generalization_min_seq_len= 10)
print(result)
