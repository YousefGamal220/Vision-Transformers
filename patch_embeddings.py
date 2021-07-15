import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, input_channels = 3, embedding_dims = 768):
        super(PatchEmbedding, self).__init__()
        '''
            image_size: the size of tha image assuming that the image is square aka height = width
            patch_size: Size of the batch assuming that it is square
            input_channel: 1 for grey_scale, 3 for RGB Channels
            embedding_dims: the dimension of the embedding layer
        '''
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedding_dims = embedding_dims
        
        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.projection = nn.Conv2d(self.input_channels, self.embedding_dims, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, x):
        # x shape: (n_samples, input_channels, image_size, image_size) -> both image_size for height and width
        projection = self.projection(x) # shape (n_samples, embedding_dim, sqrt(n_patches), sqrt(n_patches))
        projection = projection.flatten(2) # shape (n_samples, embedding_dim, n_patches)
        projection = projection.transpose(1, 2) # shape (n_samples, n_patches, embedding_dim)
        return projection

