import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool
from selfdist_toolkit.pyg_tools.stanford_ogb_utils import GINConvOGB
from selfdist_toolkit.pyg_tools.stanford_ogb_utils import AtomEncoder, BondEncoder
import torch.nn.functional as F


# todo: stochastic depth?
# todo: data augmentation? probably not at this point, right?
class GIN_basic(torch.nn.Module):
    """
    Class implementing a GIN Convolution graph wise prediction neural network. The implementation and design of this
    neural network is based on the publicly available code from Open Graph Benchmark with the code available via
    https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/
    """

    # inspired by: https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
    def __init__(
            self,
            num_classes: int = 1,
            num_layer: int = 5,
            drop_ratio: float = 0.5,
            embedding_dim: int = 300,
            atom_features: int = 9,
            bond_features: int = 3,
            jk: str = "last",
            residual: bool = False
    ):
        """
        Initialize pytorch geometric graph neural network using a number of GIN Convolution layers to derive a graph
        wise prediction of the input data. Many parameters taken from default configuration from
        https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py

        Parameters
        ----------
        num_classes : int, optional
            Number of label classes, in our case this will most certainly be 2 - one for activeness, one for
            inactiveness - using soft labels meaning output will be probability of it being active.
            Other option 1 - hard labels - either zero or one.
            Smooth labels are becoming deprecated as the active/inactive is interpreted in the OGB datasets as one
            class. (also because of pos weights in loss function), default: 1
        num_layer : int, optional
            Number of GIN Convolution layer, default: 5
        drop_ratio : float, optional
            Drop ration in the dropout functionality
        embedding_dim : int, optional
            Embedding dimension, default 300
        atom_features : int, optional
            Number of features per node=atom, in this case being 9 by default as this is the default data generation
            setting of pytorch-geometric and rdkit
        bond_features : int, optional
            Features of bond=edges. default 3
        jk :  str, optional
            Method to how to use the results of the layers of GIN Convolution: last - use last GIN Convolution output;
            sum - use the sum of layer outputs to use for the final Linear layer and then the output. Default: last
        residual : bool, optional
            Determines if the residual functionality is turned on or off. Residual means that the result of the previous
            GIN Convolution layer is added to the output of the current layer by default, treating the outcome of the
            GIN Convolution layers like a residual.
        """

        super(GIN_basic, self).__init__()

        # basic definitions
        # setting of class specific parameters for training - most of them default values
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        # data specific information
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.residual = residual
        self.jk = jk

        # GNN node related functionality
        # ==================================================================================
        # atom embedding
        self.atom_encoder = AtomEncoder(emb_dim=self.embedding_dim)
        # Convolution Layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # initialize layers
        for layer in range(self.num_layer):
            # create GIN Convolution layer using
            self.convs.append(
                GINConvOGB(emb_dim=self.embedding_dim)
            )

            # add a bach norm to the batch norm list
            self.batch_norms.append(torch.nn.BatchNorm1d(self.embedding_dim))
        # ==================================================================================

        # global pooling = pooling from all node embeddings and aggregate values. Trick is to only aggregate the nodes
        # for the current graph to predict
        self.pooling = global_mean_pool

        if self.jk == "cat":
            self.transformation_lin = torch.nn.Linear((self.num_layer+1)*self.embedding_dim, self.embedding_dim)

        # final linear layer
        # additional layer to work with the fetched node embeddings as well in the neural network context
        self.graph_pred_linear = torch.nn.Linear(self.embedding_dim, self.num_classes)

    def forward(
            self,
            batched_data: torch_geometric.data.data.Data
    ) -> torch.Tensor:
        """
        Function that is called for training and prediction (without training). When not used for training remember to
        use the pytorch function for evaluation and maybe also the annotation for no gradient usage.
        Function takes input data in form of graphs and generates predictions for them.

        Parameters
        ----------
        batched_data : torch_geometric.data.data.Data
            Graph data to use for generating the prediction

        Returns
        -------
        prediction : torch.Tensor
            Prediction generated per graph of the input data.
        """

        # GNN related code
        # ==============================================================================================
        # create embedding of the node data
        h_list = [self.atom_encoder(batched_data.x)]

        # for the convolution layer to the following steps
        for layer in range(self.num_layer):

            # generate the output of applying the convolution to the data (note that both node features and edge
            # features went through an embedding layer)
            h = self.convs[layer](x=h_list[layer], edge_index=batched_data.edge_index, edge_attr=batched_data.edge_attr)

            # transform the output using a batch norm layer
            h = self.batch_norms[layer](h)

            # if the layer is the last layer:
            if layer == self.num_layer - 1:

                # apply a dropout without using ReLU function
                h = F.dropout(h, self.drop_ratio, training=self.training)

            # in every other case
            else:

                # apply dropout in combination with relu
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            # if the residual parameter is true, then addthe result of the previous layer to this layer in a manner of
            # using a residual (with the current layer being similar to the residual of sorts)
            if self.residual:
                h += h_list[layer]

            # save the newest layer output to the list
            h_list.append(h)

        # FORK FOR THE OUTPUT GENERATION METHOD
        # Either:
        if self.jk == "last":

            # Use the last method meaning the last layer output is used for generating the prediction
            node_representation = h_list[-1]

        # or the sum of all the layers is computed
        elif self.jk == "sum":

            # initialization of something that should be treated as a zero vector initialization (I guess?)
            node_representation = 0

            # iterate over each layer and sum up results
            # comment: probably inefficient to do so, but because other routines may interfere with loss generation
            #   I will leave it as it is
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        elif self.jk == "cat":
            # concatenate all layers and push them through linear layer to reduce them to one layer again
            tmp = torch.concatenate(h_list, axis=1)

            # put through linear layer
            node_representation = self.transformation_lin(tmp)

        # ==============================================================================================
        # result to h_node
        # global node embedding/feature pooling using the batch mask which tells the algorithm which of the nodes
        # belong to which graph
        h_graph = self.pooling(node_representation, batched_data.batch)

        # Generate the output which happens by putting the pooled results through another linear layer
        return self.graph_pred_linear(h_graph)
