import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# x data (tensor), shape=(100,1)
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1,20,1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction,y)
    # 将grad重置为0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r--',lw=5)
        plt.text(0.5,0,"loss%4f"%loss.data.numpy(),fontdict={"size":20,'color':'red'})
        plt.pause(0.1)