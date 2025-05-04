import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from AlternativeModels.SequenceModels.ctgan import CTGANSynthesizer

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
    "name": "UBP paramSearch CTGAN",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Overall Score"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "generator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "discriminator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "generator_lr": {"min": 0.00001, "max": 0.001},
        "generator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_lr": {"min": 0.00001, "max": 0.001},
        "discriminator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_steps": {"min": 1, "max": 15},
        "epochs": {"min": 100, "max": 500},
        "pac": {"values": [1, 2, 4, 8, 10, 20]},
        "batch_size": {"values": [3200]}
    },
}

#sweep_id = wandb.sweep(sweep=sweep_config, project="UseCaseUBP")
sweep_id = "faruman/UseCaseUBP/pqnb8p4s"

### Priority 1
def main():
    wandb.init(project="UseCaseUBP")
    synthesizer = CTGANSynthesizer(metadata, embedding_dim= wandb.config["embedding_dim"], generator_dim= wandb.config["generator_dim"], discriminator_dim= wandb.config["discriminator_dim"],
                                    generator_lr= wandb.config["generator_lr"], generator_decay= wandb.config["generator_decay"], discriminator_lr= wandb.config["discriminator_lr"], discriminator_decay= wandb.config["discriminator_decay"], batch_size= wandb.config["batch_size"],
                                    epochs= wandb.config["epochs"], discriminator_steps= wandb.config["discriminator_steps"], pac= wandb.config["pac"],
                                   device="cuda:0", verbose=True, use_wandb=True)
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