import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
torch2array = torch_data.numpy()
data = [[1,2],[3,4]]
# change the dtype to float
tensor = torch.FloatTensor(data)
print("data",tensor)
print("matmul",torch.mm(tensor,tensor))