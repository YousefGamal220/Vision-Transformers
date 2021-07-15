import torchvision
import torch
cifar_data = torchvision.datasets.CIFAR100('/content/', download = True)
data_loader = torch.utils.data.DataLoader(cifar_data,
                                          batch_size=4,
                                          shuffle=True)