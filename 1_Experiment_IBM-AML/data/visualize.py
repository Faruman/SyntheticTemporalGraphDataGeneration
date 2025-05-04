import os

import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import seaborn as sns
from matplotlib import pyplot as plt


# get data
def create_account_nodes(account_df):
    return pd.Series({"Account": account_df["Account"].iloc[0], "Target": account_df["Target"].max(), "Timestamp": account_df["Timestamp"].min()})
data = pd.read_csv("./IBM-AML_small.csv", index_col= False)
accounts = pd.concat([data[['Sender', 'Target', 'Timestamp']].rename(columns= {"Sender":"Account"}), data[['Receiver', 'Target', 'Timestamp']].rename(columns= {"Receiver":"Account"})]).groupby("Account")[['Account', 'Target', 'Timestamp']].progress_apply(create_account_nodes)

# preview
if not os.path.exists("./plots"):
    os.mkdir("./plots")
sns.histplot(data= accounts, x= "Timestamp", hue="Target", cumulative=True, stat="density", common_norm=False)
plt.title("Distribution of Accounts over Time")
plt.savefig("./plots/acct_distribution_density.png")
plt.show()
sns.histplot(data= accounts, x= "Timestamp", hue="Target", common_norm=False)
plt.title("Distribution of Accounts over Time")
plt.savefig("./plots/acct_distribution.png")
plt.show()

sns.histplot(data= data, x= "Timestamp", hue="Target", cumulative=True, stat="density", common_norm=False)
plt.title("Distribution of Transactions over Time")
plt.savefig("./plots/trs_distribution_density.png")
plt.show()
sns.histplot(data= data, x= "Timestamp", hue="Target", common_norm=False)
plt.title("Distribution of Transactions over Time")
plt.savefig("./plots/trs_distribution.png")
plt.show()

sns.histplot(data= data, x= "Weekday")
plt.title("Distribution of Transactions per Weekday")
plt.savefig("./plots/trs_perWeekday.png")
plt.show()


