import torch
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    
    def __init__(self, num_features, num_classes, hidden_channels, num_heads, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def loss(self, pred, label):
        return torch.nn.functional.cross_entropy(pred, label)
    
    def accuracy(self, pred, label):
        correct = pred.argmax(dim=1).eq(label).sum().item()
        return correct / label.size(0)
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def __repr__(self):
        return self.__class__.__name__