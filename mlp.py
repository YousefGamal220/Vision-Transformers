# Multiple Layer Perceptron Implementation:
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_p = 0.5):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Neural Network 
        self.layer1 = nn.Linear(in_features, self.hidden_features)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(dropout_p)
    
    def forward(self, x):
        # x shape: (n_samples, n_patches + 1, in_features)
        linear1 = self.layer1(x)
        gelu = self.gelu(linear1)
        gelu = self.drop(gelu)
        linear2 = self.linear2(gelu)
        output = self.drop(linear2)
        return output