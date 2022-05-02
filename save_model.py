import torch

original = torch.load('path/to/your/checkpoint.pth')

new = {"model": original["model"]}
torch.save(new, 'path/to/new/checkpoint.pth')