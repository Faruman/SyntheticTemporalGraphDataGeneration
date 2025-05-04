import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_sparse import SparseTensor
from tqdm import trange
#adaptation of the stellargraph implementation for watchyourstep embedding generation


class AdjacencyPowerDataset(Dataset):
    """
    Dataset generating the first `num_powers` adjacency matrix powers
    and raw adjacency rows for each node given only edge_index and num_nodes.
    """
    def __init__(self, edge_index: torch.LongTensor, num_nodes: int, num_powers: int = 10, weighted: bool = False):
        """
        Args:
            edge_index (LongTensor[2, E]): Graph connectivity.
            num_nodes (int): Number of nodes in the graph.
            num_powers (int): Number of adjacency powers to compute.
            weighted (bool): If True, expects `edge_weight`; currently only unweighted supported.
        """
        super().__init__()
        self.num_powers = num_powers
        self.num_nodes = num_nodes

        # build sparse adjacency tensor
        # unweighted case: all edge weights = 1
        values = torch.ones(edge_index.size(1), dtype=torch.float)
        self.adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=values,
            sparse_sizes=(num_nodes, num_nodes),
        )

        # compute transition matrix T = D^{-1} A
        deg = self.adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        self.transition = self.adj.mul(deg_inv.view(-1, 1))

        # precompute adjacency powers: P_0 = I, P_{k+1} = T * P_k
        P = torch.eye(num_nodes)
        powers = []
        T_dense = self.transition.to_dense().T  # shape: (N, N)
        for _ in trange(num_powers, desc= "Calculate adjacency powers"):
            P = (T_dense @ P.T).T
            powers.append(P)
        self.powers = torch.stack(powers, dim=0)  # (num_powers, N, N)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx: int):
        """
        Returns:
            idx (int): the node index
            row_powers (Tensor[num_powers, num_nodes]): adjacency-powers for node idx
            adj_row (Tensor[num_nodes]): raw adjacency row for node idx
        """
        row_powers = self.powers[:, idx, :]      # (num_powers, N)
        adj_row = self.adj[idx].to_dense()       # (N,)
        return idx, row_powers, adj_row

class StreamingAdjacencyPowerDataset(Dataset):
    """
    Streamingly computes the first `num_powers` adjacency matrix powers
    and raw adjacency rows per node to avoid storing N×N×P tensors.
    """
    def __init__(self, edge_index: torch.LongTensor, num_nodes: int, num_powers: int = 10, device=None):
        super().__init__()
        self.num_powers = num_powers
        self.num_nodes = num_nodes
        self.device = device or torch.device('cpu')
        # Build sparse adjacency tensor (unweighted)
        values = torch.ones(edge_index.size(1), dtype=torch.float, device=self.device)
        self.adj = SparseTensor(
            row=edge_index[0].to(self.device),
            col=edge_index[1].to(self.device),
            value=values,
            sparse_sizes=(num_nodes, num_nodes),
        )
        # build transition matrix T = D^{-1} A as SparseTensor
        deg = self.adj.sum(dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        self.transition = self.adj.mul(deg_inv.view(-1, 1))

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx: int):
        # create one-hot vector as dense (N,)
        one_hot = torch.zeros(self.num_nodes, device=self.device)
        one_hot[idx] = 1.0
        # reshape to (N,1) for sparse matmul
        current = one_hot.unsqueeze(-1)  # shape: (N,1)
        row_powers = []
        # iteratively multiply: next = T * current
        for _ in range(self.num_powers):
            # sparse x dense -> dense (N,1)
            next_vec = self.transition.matmul(current)
            # squeeze to (N,)
            next_vec = next_vec.squeeze(-1)
            row_powers.append(next_vec)
            # prepare for next iter
            current = next_vec.unsqueeze(-1)
        # stack into (num_powers, num_nodes)
        row_powers = torch.stack(row_powers, dim=0)
        # raw adjacency row as dense 1D
        adj_row = self.adj[idx].to_dense()
        return idx, row_powers, adj_row

class AttentiveWalk(nn.Module):
    """
    Graph attention as in Watch Your Step.
    """
    def __init__(self, walk_length: int = 10):
        super().__init__()
        self.walk_length = walk_length
        self.attn = nn.Parameter(torch.empty(walk_length))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

    def forward(self, partial_powers: torch.Tensor) -> torch.Tensor:
        # partial_powers: (batch, num_powers, num_nodes)
        alpha = F.softmax(self.attn, dim=0)      # (num_powers,)
        return torch.einsum('bpn,p->bn', partial_powers, alpha)

class WatchYourStepEncoder(nn.Module):
    """
    Encoder producing node embeddings and expected-walk outputs.
    """
    def __init__(self, num_nodes: int, num_powers: int = 10,
                 embedding_dim: int = 64, num_walks: int = 80):
        super().__init__()
        if embedding_dim % 2 != 0:
            embedding_dim -= 1
        self.num_nodes = num_nodes
        self.num_powers = num_powers
        self.embedding_dim = embedding_dim
        self.num_walks = num_walks

        self.left_emb = nn.Embedding(num_nodes, embedding_dim // 2)
        self.right_lin = nn.Linear(embedding_dim // 2, num_nodes, bias=False)
        self.attentive = AttentiveWalk(walk_length=num_powers)

    def forward(self, node_idx: torch.Tensor, partial_powers: torch.Tensor) -> torch.Tensor:
        left = self.left_emb(node_idx)                   # (B, emb/2)
        right_scores = self.right_lin(left)              # (B, N)
        expected_walk = self.num_walks * self.attentive(partial_powers)
        return torch.stack([expected_walk, right_scores], dim=1)

    def embeddings(self) -> torch.Tensor:
        """
        Extract final node embeddings as numpy array of shape (num_nodes, embedding_dim).
        """
        left = self.left_emb.weight.data.cpu()           # (N, emb/2)
        right = self.right_lin.weight.data.t().cpu()     # (N, emb/2)
        return torch.cat([left, right], dim=1).numpy()

    def get_embeddings(self, node_ids) -> torch.Tensor:
        """
        Return learned embeddings for specified node indices.

        Args:
            node_ids (Tensor or list of int): indices of desired nodes.
        Returns:
            Tensor[k, embedding_dim] of embeddings for each node_id.
        """
        # ensure tensor
        if not isinstance(node_ids, torch.Tensor):
            device = self.left_emb.weight.device
            node_ids = torch.tensor(node_ids, dtype=torch.long, device=device)
        # left part
        left = self.left_emb(node_ids)                   # (k, emb/2)
        # right part
        #right = self.right_lin.weight.t()[node_ids]      # (k, emb/2)
        right = self.right_lin.weight[node_ids]
        # concatenate
        return torch.cat([left, right], dim=1)


def graph_log_likelihood(wys_output: torch.Tensor,  batch_adj: torch.Tensor) -> torch.Tensor:
    """
    Computes the Watch-Your-Step graph log-likelihood loss in PyTorch.

    Args:
        wys_output: Tensor of shape (batch_size, 2, num_nodes), where
            - channel 0 is the expected‐walks (E[r|u]),
            - channel 1 is the embedding‐outer‐product scores (u·v).
        batch_adj:   Tensor of shape (batch_size, num_nodes), containing the
                     corresponding rows of the adjacency matrix (0 or 1).

    Returns:
        A scalar loss:  sum_{i,u} | –E[r|u]·logσ(u·v)  –  (1–A[i,u])·log(1–σ(u·v)) |.
    """
    # split out the two streams
    expected_walks = wys_output[:, 0, :]    # (B, N)
    scores = wys_output[:, 1, :]    # (B, N)

    # mask where there is no edge
    adj_mask = (batch_adj == 0).float()     # (B, N)

    # log σ and log(1–σ) = logσ – scores
    log_sigmoid    = F.logsigmoid(scores)
    log1m_sigmoid  = log_sigmoid - scores

    # element-wise loss matrix
    mat = - expected_walks * log_sigmoid - adj_mask * log1m_sigmoid

    # sum absolute values over batch and nodes
    loss = mat.abs().sum()

    return loss