# ==================================================================================
# This file contains contents of the following github repository:
# https://github.com/snap-stanford/ogb/
# For more information please visit their GitHub page, for this project I will reuse part of their code in order to
# produce a similar model of GIN graph wise prediction as a base model. There will be modifications regarding adaptation
# to my data structure.
# ==================================================================================
import torch


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, num_embeddings):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(num_embeddings):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, num_embeddings):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(num_embeddings):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
