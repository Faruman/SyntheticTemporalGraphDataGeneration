import numpy as np
import pandas as pd
from typing import List

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from sdv.metadata import SingleTableMetadata

import os
import pickle
import joblib


class GraphSynthModel:
    def __init__(self, node_embedding_dim= 2, device= None, verbose= False):
        self.node_embedding_dim = node_embedding_dim

        self.data_preprocessed = False
        self.trained = False

        self.edge_index_cols = []

        self.embeddingGenerator = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.verbose = verbose

    def fit_data_preprocessing(self, data: pd.DataFrame, edge_index_cols: List[str], watchYourStep_epochs= 100, watchYourStep_patience= 3, watchYourStep_batchSize= 128, watchYourStep_lr= 1e-3):
        data = data.copy(deep=True)

    def fit_synth_model(self, model, model_parameters, data, edge_index_cols, model_type= "sdv"):
        data = data.copy(deep=True)

        return self.model

    def import_synth_model(self, model):
        self.model = model

        self.trained = True

    def sample(self, num_samples, kmeans_batchSize= 6400):
        assert self.trained, "Please train the model before sampling data"

        synthetic_data = self.model.sample(num_samples)
        return synthetic_data

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

    def load(self, path: str):
        # Load metadata
        pass
