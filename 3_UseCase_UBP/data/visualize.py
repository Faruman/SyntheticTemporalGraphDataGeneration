import os

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt


data = pd.read_csv("./UBP.csv", index_col= False)

#preview
account_cols = ['PCACCT1', 'PCACCT2', 'PCACCT3', 'PCACCT4', 'PCACCT5', 'PCACCT6', 'PCACCT7', 'PCACCT8', 'PCACCT9', 'PCACCT10', 'PCACCT11', 'PCACCT12', 'PCACCT13', 'PCACCT14', 'PCACCT15', 'PCACCT16', 'PCACCT17', 'PCACCT18', 'PCACCT19', 'PCACCT20', 'PCACCT21', 'PCACCT22', 'PCACCT23']
def create_account_nodes(account_df):
    return pd.Series({"account": account_df["account"].iloc[0], "fraud": account_df["fraud"].max(), "timestamp": account_df["timestamp"].min()})
print(data.head())
accounts = pd.concat([data[['source_id', 'fraud', 'timestamp']].rename(columns= {"source_id":"account"}), data[['target_id', 'fraud', 'timestamp']].rename(columns= {"target_id":"account"})]).groupby("account")[['account', 'fraud', 'timestamp']].apply(create_account_nodes)

if not os.path.exists("./plots"):
    os.mkdir("./plots")

sns.histplot(data= accounts, x= "timestamp", hue="fraud", cumulative=True, stat="density", common_norm=False)
plt.title("Distribution of Accounts over Time")
plt.savefig("./plots/acct_distribution_density.png")
plt.show()
sns.histplot(data= accounts, x= "timestamp", hue="fraud", common_norm=False)
plt.title("Distribution of Accounts over Time")
plt.savefig("./plots/acct_distribution.png")
plt.show()

sns.histplot(data= data, x= "timestamp", hue="fraud", cumulative=True, stat="density", common_norm=False)
plt.title("Distribution of Transactions over Time")
plt.savefig("./plots/trs_distribution_density.png")
plt.show()
sns.histplot(data= data, x= "timestamp", hue="fraud", common_norm=False)
plt.title("Distribution of Transactions over Time")
plt.savefig("./plots/trs_distribution.png")
plt.show()