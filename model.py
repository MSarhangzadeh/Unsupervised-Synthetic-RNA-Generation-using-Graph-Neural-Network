import torch 
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from gcn import GCNConv

class GCNModel(nn.Module):
    
    """
    A two-layer Graph Convolutional Network (GCN) model for node classification.
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_dim (int): Number of hidden units in the first GCN layer.
        out_channels (int): Number of classes or output features.

    Forward Inputs:
        x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
        edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

    Returns:
        Tensor: Log-probabilities for each class per node with shape [num_nodes, out_channels].
    """
    
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


#for testing
'''
def test_gcn_encoder(num_nodes, num_node_features, hidden_dim, out_channels, num_edges):
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    model = GCNModel(in_channels=num_node_features, hidden_dim=hidden_dim, out_channels=out_channels)
    out = model(x, edge_index)  
    
    return out.shape

#32 nodes, 5 features, output dim 64, 62 edges
test_out_shape = test_gcn_encoder(num_nodes=32, num_node_features=5, hidden_dim=32, out_channels=64, num_edges=62)
print(test_out_shape)  # Expect torch.Size([1, 64])
'''
