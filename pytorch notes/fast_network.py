import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# data
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor) # 32 bit floating
y = torch.cat((y0,y1),).type(torch.LongTensor) # 64 bit integer
x,y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
# plt.show()

net = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(net)
plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.0
        plt.text(1.5,-4,'Accuracy=%.2f' % accuracy,fontdict={"size":20,"color":"red",})
        plt.pause(0.1)
plt.ioff()
plt.show()