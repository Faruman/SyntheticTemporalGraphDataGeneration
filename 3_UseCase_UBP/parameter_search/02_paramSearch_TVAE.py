import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from AlternativeModels.SequenceModels.tvae import TVAESynthesizer

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
    "name": "UBP paramSearch TVAE",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Overall Score"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "compress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "decompress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "l2scale": {"min": 0.00001, "max": 0.001},
        "loss_factor": {"min": 1, "max": 5},
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="UseCaseUBP")
#sweep_id = "financialDataGeneration/FinancialDataGeneration_TVAE_ParamSearch/bh90ig0h"

### Priority 1
def main():
    wandb.init(project="UseCaseUBP")
    synthesizer = TVAESynthesizer(metadata, embedding_dim= wandb.config["embedding_dim"], compress_dims= wandb.config["compress_dims"], decompress_dims= wandb.config["decompress_dims"],
                                    l2scale= wandb.config["l2scale"], loss_factor= wandb.config["loss_factor"], learning_rate= wandb.config["learning_rate"],
                                    epochs= wandb.config["epochs"], batch_size= wandb.config["batch_size"],
                                  device= "cuda:1",verbose=True, use_wandb=True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=len(real_data) - 1)

    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict())
    quality_score = quality_report.get_score()

    privacy_score = DCRBaselineProtection.compute(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata.to_dict(),
                                                  num_rows_subsample=5000, num_iterations=3)

    score = privacy_score * quality_score

    wandb.log({**quality_report.get_properties().set_index("Property")["Score"].to_dict(), **{"Quality Score": quality_score, "Privacy Score": privacy_score, "Overall Score": score}})
    wandb.finish()


wandb.agent(sweep_id, function=main, count=30)