import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(2)

# fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # tensor shape = (100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size()) # noisy y data shape (100,1)
# x,y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # plot result
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title("NET1")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

    torch.save(net1,"net.pkl") # entire net
    torch.save(net1.state_dict(),"net_parameters.pkl") # 保存神经元参数

def restore_net():
    net2 = torch.load("net.pkl")
    prediction = net2(x)

    plt.subplot(132)
    plt.title("NET2")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load("net_parameters.pkl"))
    prediction = net3(x)
    plt.subplot(133)
    plt.title("NET3")
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    plt.show()

# save model
save()
# load model
restore_net()
# load only parameters
restore_params()