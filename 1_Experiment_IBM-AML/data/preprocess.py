import os

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA


data = pd.read_csv('./raw/HI-Medium_Trans.csv')

# preprocess dataset
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["Weekday"] = data["Timestamp"].dt.weekday
data["Timestamp"] = data["Timestamp"].astype('int64') // 10**9

data["Sender"] = data["From Bank"].astype(str) + "_" + data["Account"].astype(str)
data["Receiver"] = data["To Bank"].astype(str) + "_" + data["Account.1"].astype(str)
data = data.drop(columns=["From Bank", "To Bank", "Account", "Account.1"])

data = data.rename(columns={"Is Laundering": "Target", "Payment Currency": "Sending Currency"})

data = data.dropna(subset= ["Sender", "Receiver"])

print("Original Fraud Ratio: {}".format(data["Target"].sum()/len(data)))

# Filter top 5 currencies
top_n_currencies = 5
data = data.loc[data["Sending Currency"].str.contains('|'.join(list(data["Sending Currency"].value_counts().index[:top_n_currencies])))]
data = data.loc[data["Receiving Currency"].str.contains('|'.join(list(data["Receiving Currency"].value_counts().index[:top_n_currencies])))]
data = data.loc[data["Sender"] != data["Receiver"]]
print("Post-Currency-Filtering Fraud Ratio: {}".format(data["Target"].sum()/len(data)))

min_transactions_per_node = 3
max_transactions_per_node = 300
valid_accounts = pd.concat([data["Sender"], data["Receiver"]]).value_counts().reset_index()
valid_accounts = valid_accounts.loc[valid_accounts["count"] > min_transactions_per_node]
valid_accounts = valid_accounts["index"]
data = data.loc[data["Receiver"].isin(valid_accounts) & data["Sender"].isin(valid_accounts)]
print("Post-MinAccount-Filtering Fraud Ratio: {}".format(data["Target"].sum()/len(data)))

hist, bins = np.histogram(data["Timestamp"].values, bins=100)
bins_exclude = hist < len(data)*0.0001
start_bin_idx = 0
for bin_bool in bins_exclude:
    if bin_bool:
        start_bin_idx += 1
    else:
        break
end_bin_idx = 0
for bin_bool in bins_exclude[::-1]:
    if bin_bool:
        end_bin_idx += 1
    else:
        break
data = data.loc[(data["Timestamp"] > bins[start_bin_idx]) & (data["Timestamp"] < bins[-(end_bin_idx-1)])]
print("Post-EmptyRange-Filtering Fraud Ratio: {}".format(data["Target"].sum()/len(data)))

payForm_cols = data["Payment Format"].unique()
curr_cols = pd.Series(data[["Sending Currency", "Receiving Currency"]].values.flatten()).unique()

def create_node_data(group, id):
    id = pd.DataFrame([group[id].unique()[0]], index= [0], columns= ["ID"])
    payForm_count = pd.DataFrame(group["Payment Format"].value_counts().to_dict(), index= [0], columns= payForm_cols).fillna(0)
    payForm_count = payForm_count / np.sum(payForm_count.values)
    curr_count = pd.DataFrame((group["Sending Currency"].value_counts() + group["Receiving Currency"].value_counts()).to_dict(),  index= [0], columns= curr_cols).fillna(0)
    curr_count = curr_count / np.sum(curr_count.values)
    if (group["Timestamp"].max() - group["Timestamp"].min()) > 0:
        activity = pd.DataFrame([group["Amount Paid"].sum() / (group["Timestamp"].max() - group["Timestamp"].min())], index= [0], columns= ["Activity"])
    else:
        activity = pd.DataFrame([0], index=[0], columns=["Activity"])
    return pd.concat([id, payForm_count, curr_count, activity], axis= 1).iloc[0]

sender_df = data.groupby("Sender").progress_apply(lambda x: create_node_data(x, "Sender")).add_prefix("Sender_").reset_index(drop=True)
receiver_df = data.groupby("Receiver").progress_apply(lambda x: create_node_data(x, "Receiver")).add_prefix("Receiver_").reset_index(drop=True)
account_df = pd.merge(sender_df, receiver_df, how= "outer", left_on= "Sender_ID", right_on= "Receiver_ID")
id_df = account_df[["Sender_ID", "Receiver_ID"]].apply(lambda x: x["Sender_ID"] if isinstance(x["Sender_ID"], str) else x["Receiver_ID"], axis= 1)
account_df = account_df[[col for col in account_df.columns if col not in ["Sender_ID", "Receiver_ID"]]].fillna(account_df[[col for col in account_df.columns if col not in ["Sender_ID", "Receiver_ID"]]].mean())
#account_df = account_df[[col for col in account_df.columns if col not in ["Sender_ID", "Receiver_ID"]]].fillna(0)

num_pca= 3
pca = PCA(n_components=num_pca)
pca.fit(account_df.values)
print("Retained account feature variance: {}".format(sum(pca.explained_variance_ratio_)))
account_df = pd.DataFrame(pca.transform(account_df.values), columns= ["PCA{}".format(i) for i in range(num_pca)])
account_df = pd.concat((id_df, account_df), axis= 1).rename(columns= {0: "ID"})

data = pd.merge(data, account_df.rename(columns= dict(zip(account_df.columns, ["Sender_{}".format(x) for x in account_df.columns]))), how= "left", left_on= "Sender", right_on= "Sender_ID").drop(columns= ["Sender_ID"])
data = pd.merge(data, account_df.rename(columns= dict(zip(account_df.columns, ["Receiver_{}".format(x) for x in account_df.columns]))), how= "left", left_on= "Receiver", right_on= "Receiver_ID").drop(columns= ["Receiver_ID"])
#temp = data.groupby("Receiver").progress_apply(lambda x: create_node_data(x, "Receiver")).add_prefix("Receiver_").reset_index(drop=True)
#data = pd.merge(data, temp, how= "left", left_on= "Receiver", right_on= "Receiver_ID").drop(columns= ["Receiver_ID"])

for column in data.columns:
    dtype = data[column].dtype.name
    if dtype == "object" and column not in ["Sender", "Receiver"]:
        oh = OneHotEncoder()
        temp = oh.fit_transform(data[[column]])
        data = pd.concat([data, pd.DataFrame(temp.toarray(), columns= ["{}_{}".format(column, i) for i in range(temp.shape[1])], index= data.index)], axis=1)
        data = data.drop(columns=[column])

def create_account_nodes(account_df):
    return pd.Series({"Account": account_df["Account"].iloc[0], "Target": account_df["Target"].max(), "Timestamp": account_df["Timestamp"].min()})
accounts = pd.concat([data[['Sender', 'Target', 'Timestamp']].rename(columns= {"Sender":"Account"}), data[['Receiver', 'Target', 'Timestamp']].rename(columns= {"Receiver":"Account"})]).groupby("Account")[['Account', 'Target', 'Timestamp']].progress_apply(create_account_nodes)
od = OrdinalEncoder()
od.fit(accounts[["Account"]].values)
data["Sender"] = od.transform(data[["Sender"]].values)
data["Receiver"] = od.transform(data[["Receiver"]].values)

#statistics
print("Total Cases: {}".format(data.shape[0]))
print("Fraudulent Cases: {}".format(data["Target"].sum()))
print("Transactions per Sender: {}".format(data["Sender"].value_counts().mean()))
print("Transactions per Receiver: {}".format(data["Receiver"].value_counts().mean()))

print(data.head())

data.to_csv("./IBM-AML.csv", index= False)