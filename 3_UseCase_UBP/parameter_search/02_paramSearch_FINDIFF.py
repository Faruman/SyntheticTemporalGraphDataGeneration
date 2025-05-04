import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from AlternativeModels.SequenceModels.findiff import FINDIFFSynthesizer

from sdmetrics.single_table import DCRBaselineProtection, DisclosureProtectionEstimate
from sdmetrics.reports.single_table import QualityReport

import wandb

## setup wandb
#wandb.login()

real_data = pd.read_csv("./temp/train_data.csv", index_col=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

## Test CTGAN
sweep_config = {
    "name": "UBP paramSearch FinDiff",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Overall Score"},
    "parameters": {
        "cat_embedding_dim": {"values": [2, 4, 8]},
        "mlp_dim": {"values": [(512, 512, 512), (1024, 1024, 1024, 1024), (2048, 2048, 2048)]},
        "mlp_activation": {"values": ["lrelu", "relu", "tanh"]},
        "diffusion_steps": {"min": 200, "max": 1000},
        "diffusion_beta_start": {"min": 0.00001, "max": 0.001},
        "diffusion_beta_end": {"min": 0.001, "max": 0.1},
        "mlp_lr": {"min": 0.00001, "max": 0.001},
        "epochs": {"min": 100, "max": 500},
        "batch_size": {"values": [3200]}
    },
}

#sweep_id = wandb.sweep(sweep=sweep_config, project="UseCaseUBP")
sweep_id = "faruman/UseCaseUBP/r0jglc2p"

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_FINDIFF_ParamSearch")
    synthesizer = FINDIFFSynthesizer(metadata, cat_embedding_dim= wandb.config["cat_embedding_dim"], mlp_dim= wandb.config["mlp_dim"], mlp_activation= wandb.config["mlp_activation"],
                                    diffusion_steps= wandb.config["diffusion_steps"], diffusion_beta_start= wandb.config["diffusion_beta_start"], diffusion_beta_end= wandb.config["diffusion_beta_end"],
                                    mlp_lr=wandb.config["mlp_lr"], epochs= wandb.config["epochs"], batch_size= wandb.config["batch_size"],
                                     device="cuda:0",  verbose=True, use_wandb=True)

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