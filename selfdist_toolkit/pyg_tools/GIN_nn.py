import torch
from torch_geometric.nn import global_mean_pool, GINConv
from selfdist_toolkit.pyg_tools.stanford_ogb_utils import AtomEncoder, BondEncoder
import torch.nn.functional as F


class GIN_basic(torch.nn.Module):
    # inspired by: https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
    def __init__(self, num_classes):
        super(GIN_basic, self).__init()

        # basic definitions
        self.num_layer = 5
        self.drop_ratio = 0.5
        self.embedding_dim = 300
        self.num_classes = num_classes
        # data specific information
        self.atom_features = 9
        self.bond_features = 3


        # GNN node related functionality
        # ==================================================================================
        # atom embedding
        self.atom_encoder = AtomEncoder(emb_dim=self.embedding_dim, num_embeddings=self.atom_features)
        # bond embedding
        self.bond_encoder = BondEncoder(emb_dim=self.embedding_dim, num_embeddings=self.bond_features)
        # Convolution Layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual = False
        self.JK = "last"

        # initialize layers
        for layer in range(self.num_layer):
            self.convs.append(GINConv(
                nn=torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_dim, 2*self.embedding_dim),
                    torch.nn.BatchNorm1d(2*self.embedding_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*self.embedding_dim, self.embedding_dim)
                )
            ))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.embedding_dim))

        # ==================================================================================

        # pooling
        self.pooling = global_mean_pool

        # final linear layer
        self.graph_pred_linear = torch.nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, batched_data):
        # gnn node stuff

        # GNN related code
        # ==============================================================================================
        h_list = [self.atom_encoder(batched_data.x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], batched_data.edge_index, batched_data.edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        # ==============================================================================================
        # result to h_node
        # pooling
        h_graph = self.pool(node_representation, batched_data.batch)

        # graph pred linear
        return self.graph_pred_linear(h_graph)



