import torch.nn as nn
class AttentionModel(nn.Module):
    def __init__(self, dim, num_heads, include_bias, attention_dropout = 0.5, projection_dropout = 0.5):
        super(AttentionModel, self).__init__()
        '''
            dim: Input/Output dimensions
            num_heads: number of heads of the attention
            include_bias: bool variable to include bias or not for query, key, and value of the attention
            attention_dropout: probability of dropout for the attention 
            projection_dropout: robability of dropout for the projection (Patch Embedding Layer)
        '''
        self.dim = dim
        self.num_heads = num_heads
        self.include_bias = include_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_layer = nn.Linear(dim, dim * 3, bias = include_bias) # Linear Mapping take in token embedding and generate query, key and a value (reason for dim * 3)
        self.projection = nn.Linear(dim, dim)

        self.attention_drop = nn.Dropout(self.attention_dropout)
        self.projection_drop = nn.Dropout(self.projection_dropout)
    
    def forward(self, x):
        # x shape: (n_samples, n_patches + 1, dim) [num_patches +1 for the 0 class token (from the paper)]
        
        # Extract the dimensions:
        n_samples, n_tokens, dim = x.shape
        linear = self.linear_layer(x) # shape: (n_samples, n_patches + 1, dim * 3)
        linear = linear.reshape(n_samples, n_tokens, 3, self.num_heads, self.head_dim) # shape: (n_samples, n_tokens, 3, num_heads, head_dim) 
        linear = linear.permute(2, 0, 3, 1, 4) # shape: (3, n_samples, num_heads,  n_patches + 1, head_dim) # To Extract query, key, value
        query = linear[0]
        key = linear[1]
        value = linear[2]

        key_transpose = key.transpose(-2, -1) # Shape (num_samples, num_heads, head_dim, n_patches + 1)
        query_key = (query @ key_transpose) * self.scale # From Attention all you Need [Transformers]
        attention = query_key.softmax(dim = -1) # (n_samples, n_heads, n_patches + 1, ) To Generate a discrete probability distribution that sums up to one for [weighted average]
        attention = self.attention_drop(attention)
        weighted_average = attention @ value
        weighted_average_transpose = weighted_average.transpose(1, 2)
        weighted_average_flat = weighted_average_transpose.flatten(2) # To Flat the last 2 dimensions [For concatination] shape:(n_samples, n_patches + 1, head_dim)
        output = self.projection(weighted_average_flat) # shape: (n_samples, n_patches+1, dim)
        output = self.projection_drop(output)

        return output
