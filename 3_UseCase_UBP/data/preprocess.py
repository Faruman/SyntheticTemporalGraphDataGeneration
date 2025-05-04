import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


data = pd.read_csv('./raw/transformed_pca_sample_v2_2025-02-26.csv', index_col=0)

# preprocess dataset
def convert_time(timestamp):
    part1 = int(timestamp.split(':')[0])
    part2 = float(timestamp.split(':')[1])
    return part1 * 3600 + part2 * 60
data["timestamp"] = data["transfer_datetime"].apply(convert_time)
data = data.drop(columns=['transfer_datetime'])
data["timestamp"] = (data["timestamp"] - data["timestamp"].min())

print("Total Cases: {}".format(data.shape[0]))
print("Fraudulent Cases: {}".format(data["fraud"].sum()))
print("Transactions per Sender: {}".format(data["source_id"].value_counts().mean()))
print("Transactions per Receiver: {}".format(data["target_id"].value_counts().mean()))

data.to_csv("./UBP.csv", index= False)

