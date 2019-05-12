import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10) # 10 points from 1 to 10
y = torch.linspace(10,1,10) # 10 points from 10 to 1

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # random shuffle for training
    # num_workers=2, # 设置为两个线程或两个进程进行提取，使提取更有效率
    # 在编辑器里运行时会因为多线程而报错，在terminal内运行则无此类问题
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print("Epoch:",epoch,'\t Step:',step,"\t batch x:",batch_x.numpy(),"\t batch y:",batch_y.numpy())