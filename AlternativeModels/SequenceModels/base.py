import numpy as np
import pandas as pd
from typing import List

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .graph_encoder import WatchYourStepEncoder, AdjacencyPowerDataset, StreamingAdjacencyPowerDataset, graph_log_likelihood

from sdv.metadata import SingleTableMetadata

import os
import pickle
import joblib


class SeqSynthModel:
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

        self.edge_index_cols = edge_index_cols
        self.num_nodes = pd.concat([data[col] for col in edge_index_cols]).nunique()
        self.node_relabel_dict = dict(zip(pd.concat([data[col] for col in edge_index_cols]).unique(), range(self.num_nodes)))
        for col in edge_index_cols:
            data[col] = data[col].map(self.node_relabel_dict)

        self.real_nodes_per_datapoint = self.num_nodes / (data.shape[0]*2)

        # 2) build undirected edge_index
        src = torch.tensor(data[edge_index_cols[0]].values, dtype=torch.long)
        dst = torch.tensor(data[edge_index_cols[1]].values, dtype=torch.long)
        edge_index = torch.cat(
            [torch.stack([src, dst], dim=0),
             torch.stack([dst, src], dim=0)],
            dim=1
        )

        # 3) dataset + split for early‐stop
        full_ds = StreamingAdjacencyPowerDataset(
            edge_index=edge_index,
            num_nodes=self.num_nodes,
            num_powers=10,
        )
        val_size = int(0.1 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=watchYourStep_batchSize, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=watchYourStep_batchSize, shuffle=False)

        # 4) model, optimizer

        model = WatchYourStepEncoder(
            num_nodes=self.num_nodes,
            num_powers=10,
            embedding_dim=self.node_embedding_dim,
            num_walks=80,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=watchYourStep_lr)

        # 5) training with early stopping
        best_val = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, watchYourStep_epochs + 1):
            model.train()
            total_loss = 0.0
            for idxs, powers, adj_row in tqdm(train_loader, desc= "Training WatchYourStep (epoch: {})".format(epoch)):
                idxs = idxs.to(self.device)
                powers = powers.to(self.device)
                adj_row = adj_row.to(self.device)

                out = model(idxs, powers)  # (B,2,N)
                loss = graph_log_likelihood(out, adj_row)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idxs, powers, adj_row in tqdm(val_loader, desc= "Validating WatchYourStep (epoch: {})".format(epoch)):
                    idxs = idxs.to(self.device)
                    powers = powers.to(self.device)
                    adj_row = adj_row.to(self.device)
                    out = model(idxs, powers)
                    val_loss += graph_log_likelihood(out, adj_row).item()

            val_loss /= len(val_loader)
            if self.verbose:
                print(f"Epoch {epoch}: train_loss={total_loss / len(train_loader):.4f}, val_loss={val_loss:.4f}")

            # check improvement
            if val_loss * 1.01 < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= watchYourStep_patience:
                    if self.verbose:
                        print(f"No improvement for {watchYourStep_patience} epochs - stopping at epoch {epoch}.")
                    break

        # restore best
        model.load_state_dict(best_state)

        # 6) extract embeddings
        model.eval()
        with torch.no_grad():
            all_idx = torch.arange(self.num_nodes, device= self.device)
            embs = model.get_embeddings(all_idx).cpu().numpy()

        self.embeddings = embs  # shape: (num_nodes, embedding_dim)
        self.data_preprocessed = True

    def fit_synth_model(self, model, model_parameters, data, edge_index_cols, model_type= "sdv"):
        data = data.copy(deep=True)

        assert self.data_preprocessed, "Data is not preprocessed, run fit_data_preprocess() first"

        for col in edge_index_cols:
            data[col] = data[col].map(self.node_relabel_dict)
            data = pd.concat((data, pd.DataFrame(self.embeddings[data[col].values], columns=[f"{col}_WYS_{i}" for i in range(self.node_embedding_dim)])), axis=1)
            data = data.drop(columns=[col])

        if "context_columns" in model_parameters:
            model_parameters["context_columns"] = model_parameters["context_columns"] + [f"{col}_WYS_{i}" for i in range(self.node_embedding_dim) for col in edge_index_cols]

        if model_type == "sdv":
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
            self.model = model(metadata, **model_parameters)
        else:
            self.model = model(**model_parameters)

        self.model.fit(data=data)

        self.trained = True
        return self.model

    def import_synth_model(self, model):
        self.model = model

        self.trained = True

    def sample(self, num_samples, kmeans_batchSize= 6400):
        assert self.trained, "Please train the model before sampling data"

        synthetic_data = self.model.sample(num_samples)

        synthetic_nodes = []
        for col in self.edge_index_cols:
            synthetic_nodes.append(synthetic_data[[f"{col}_WYS_{i}" for i in range(self.node_embedding_dim)]].rename(columns=dict(zip([f"{col}_WYS_{i}" for i in range(self.node_embedding_dim)], [f"WYS_{i}" for i in range(self.node_embedding_dim)]))))
        synthetic_nodes = pd.concat(synthetic_nodes, axis= 0).drop_duplicates()
        scaler = StandardScaler()
        synthetic_nodes_scaled = pd.DataFrame(scaler.fit_transform(synthetic_nodes), columns=synthetic_nodes.columns)
        n_clusters = int(self.real_nodes_per_datapoint * synthetic_data.shape[0])
        kms = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", batch_size= kmeans_batchSize, verbose= self.verbose)
        kms.fit(synthetic_nodes_scaled)
        synthetic_nodes["node_id"] = kms.predict(synthetic_nodes_scaled)

        for col in self.edge_index_cols:
            synthetic_data = pd.merge(synthetic_data, synthetic_nodes, left_on= [f"{col}_WYS_{i}" for i in range(self.node_embedding_dim)], right_on= [f"WYS_{i}" for i in range(self.node_embedding_dim)], how= "left").rename(columns= {"node_id": col})
            synthetic_data = synthetic_data.drop(columns= [f"{col}_WYS_{i}" for i in range(self.node_embedding_dim)] + [f"WYS_{i}" for i in range(self.node_embedding_dim)])

        return synthetic_data

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        # Save metadata
        meta = {
            "node_embedding_dim": self.node_embedding_dim,
            "data_preprocessed": self.data_preprocessed,
            "trained": self.trained,
            "edge_index_cols": self.edge_index_cols,
            "embeddings": self.embeddings,
            "node_relabel_dict": self.node_relabel_dict,
            "real_nodes_per_datapoint": self.real_nodes_per_datapoint,
            "verbose": self.verbose
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str):
        # Load metadata
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.node_embedding_dim = meta["node_embedding_dim"]
        self.data_preprocessed = meta["data_preprocessed"]
        self.trained = meta["trained"]
        self.edge_index_cols = meta["edge_index_cols"]
        self.embeddings = meta["embeddings"]
        self.node_relabel_dict = meta.get("node_relabel_dict")
        self.real_nodes_per_datapoint = meta["real_nodes_per_datapoint"]
        self.verbose = meta["verbose"]
