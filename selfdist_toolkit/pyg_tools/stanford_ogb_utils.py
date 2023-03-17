# ==================================================================================
# This file contains contents of the following github repository:
# https://github.com/snap-stanford/ogb/
# For more information please visit their GitHub page, for this project I will reuse part of their code in order to
# produce a similar model of GIN graph wise prediction as a base model. There will be modifications regarding adaptation
# to my data structure.
# ==================================================================================
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import smiles

# hardcoded info from ogb file - DEPRECATED AS PYTORCH PRODUCES DIFFERENT FEATURES
# num_embeddings_atom = [119, 5, 12, 12, 10, 6, 6, 2, 2]
# num_embeddings_bond = [5, 6, 2]

# new embedding dimensions from pytorch geometric
num_embeddings_atom = list(map(len, smiles.x_map.values()))
num_embeddings_bond = list(map(len, smiles.e_map.values()))


class GINConvOGB(MessagePassing):
    """
    Original source: https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
    """
    def __init__(
            self,
            emb_dim: int
    ):
        """
        Initializing function of GINConvOGB.

        Parameters
        ----------
        emb_dim : int
            node embedding dimensionality - dimension the input data is stretched or compressed into
        """

        # super class method call
        super(GINConvOGB, self).__init__(aggr="add")

        # initialize machine learning object just like in GINConv from pytorch geometric
        self.mlp = torch.nn.Sequential(
            # linear layer
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            # norm
            torch.nn.BatchNorm1d(2 * emb_dim),
            # relu
            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))

        # parameter for eps learning
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # bond encoder to transfer edge attribute (and node attributes) to a common dimension
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Function that implements GIN convolution functionality with the addition of using edge attributes

        Parameters
        ----------
        x : torch.Tensor
            input node attributes/embedding
        edge_index : torch.Tensor
            input edge indexes
        edge_attr : torch.Tensor
            input edge attributes

        Returns
        -------
        out : torch.Tensor
            result of applying GIN Convolution using edge attributes
        """

        # generate edge embedding (same dimension as node embedding)
        edge_embedding = self.bond_encoder(edge_attr)

        # calculate the output using GIN Convolution logic with the addition of using edge attributes
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        # return result
        return out

    def message(
            self,
            x_j,
            edge_attr
    ):
        return torch.nn.functional.relu(x_j + edge_attr)

    def update(
            self,
            aggr_out
    ):
        return aggr_out


class AtomEncoder(torch.nn.Module):
    """
    Original source:
    https://github.com/snap-stanford/ogb/blob/d37cffa2e2cde531ca7b7e75800d331ed1e738a6/ogb/graphproppred/mol_encoder.py
    """

    def __init__(
            self,
            emb_dim
    ):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for dim in num_embeddings_atom:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(
            self,
            x
    ):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    """
    Original source:
    https://github.com/snap-stanford/ogb/blob/d37cffa2e2cde531ca7b7e75800d331ed1e738a6/ogb/graphproppred/mol_encoder.py
    """

    def __init__(
            self,
            emb_dim
    ):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(num_embeddings_bond):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(
            self,
            edge_attr
    ):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
