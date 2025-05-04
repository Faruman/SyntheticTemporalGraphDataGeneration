import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from AlternativeModels.SequenceModels.timegan import TIMEGANSynthesizer

from sdmetrics.single_table import DCRBaselineProtection, DisclosureProtectionEstimate
from sdmetrics.reports.single_table import QualityReport

import wandb

## setup wandb
#wandb.login()

min_number_edges_per_node = 2
context_columns = ['PCACCT1', 'PCACCT2', 'PCACCT3', 'PCACCT4', 'PCACCT5', 'PCACCT6',
       'PCACCT7', 'PCACCT8', 'PCACCT9', 'PCACCT10', 'PCACCT11', 'PCACCT12',
       'PCACCT13', 'PCACCT14', 'PCACCT15', 'PCACCT16', 'PCACCT17', 'PCACCT18',
       'PCACCT19', 'PCACCT20', 'PCACCT21', 'PCACCT22', 'PCACCT23', 'source_id_WYS_0',
       'source_id_WYS_1', 'source_id_WYS_2', 'source_id_WYS_3',
       'source_id_WYS_4', 'source_id_WYS_5']

## load data
real_data = pd.read_csv("./temp/train_data.csv", index_col=False)

real_data["timestamp"] = pd.to_datetime(real_data["timestamp"]).astype('int64') // 10**9

mi = pd.MultiIndex.from_frame(real_data[context_columns])
real_data['source_id'] = pd.factorize(mi)[0]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='source_id', sdtype='id')
metadata.update_column(column_name='timestamp', sdtype='numerical')
metadata.set_sequence_key(column_name='source_id')
metadata.set_sequence_index(column_name='timestamp')
metadata.set_primary_key(None)

## Truncate sequences
def truncate_sequence(group, max_len, min_len, id_column):
    if len(group) <= max_len and len(group) >= min_len:
        group[id_column] = group[id_column].apply(lambda x: f"{x}_0")
        return group
    elif len(group) > max_len:
        out = pd.DataFrame(columns=group.columns)
        for i in range(len(group) // max_len):
            seq = group.sample(min(len(group), max_len))
            seq[id_column] = seq[id_column].apply(lambda x: f"{x}_{i}")
            if out.empty:
                out = seq
            else:
                out = pd.concat((out, seq))
            group = group.drop(seq.index)
        return out
    else:
        return pd.DataFrame(columns=group.columns)
tqdm.pandas(desc="Truncating sequences", leave=True)
real_data = real_data.groupby(context_columns).progress_apply(truncate_sequence, max_len= 30, min_len= min_number_edges_per_node, id_column= "source_id").reset_index(drop=True)

## Test CTGAN
sweep_config = {
    "name": "UBP paramSearch TIMEGAN",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Overall Score"},
    "parameters": {
        "hidden_dim": {"values": [16, 32, 64,  256]},
        "num_layers": {"values": [2, 3, 4]},
        "metric_iteration": {"values": [5, 10, 15]},
        "beta1": {"min": 0.2, "max": 1.0},
        "learning_rate": {"min": 0.00001, "max": 0.005},
        "gamma": {"min": 0.1, "max": 10.0},
        "encoder_loss_weight_s": {"min": 0.01, "max": 1.0},
        "encoder_loss_weight_0": {"min": 0.5, "max": 50.0},
        "generator_loss_weight": {"min": 50, "max": 150},
        "generator_steps": {"min": 1, "max": 5},
        "epochs": {"min": 100, "max": 500},
        "batch_size": {"values": [256]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="UseCaseUBP")
#sweep_id = "faruman/UseCaseUBP/5bkul5b2"

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_TIMEGAN_ParamSearch")
    synthesizer = TIMEGANSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, hidden_dim= wandb.config["hidden_dim"],
                                     num_layers= wandb.config["num_layers"], metric_iteration= wandb.config["metric_iteration"], beta1= wandb.config["beta1"],learning_rate= wandb.config["learning_rate"], gamma= wandb.config["gamma"],
                                     encoder_loss_weight_s= wandb.config["encoder_loss_weight_s"], encoder_loss_weight_0= wandb.config["encoder_loss_weight_0"], generator_loss_weight= wandb.config["generator_loss_weight"],
                                     generator_steps= wandb.config["generator_steps"], epochs= wandb.config["epochs"], batch_size= wandb.config["batch_size"],
                                     device="cuda:0", verbose= True, use_wandb= True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=len(real_data)-1)

    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict())
    quality_score = quality_report.get_score()

    privacy_score = DCRBaselineProtection.compute(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata.to_dict(), num_rows_subsample= 5000, num_iterations= 3)

    score = privacy_score * quality_score

    wandb.log({**quality_report.get_properties().set_index("Property")["Score"].to_dict(), **{"Quality Score": quality_score, "Privacy Score": privacy_score, "Overall Score": score}})
    wandb.finish()

wandb.agent(sweep_id, function=main, count=30)