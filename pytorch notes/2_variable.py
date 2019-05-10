import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
# require gred 定义是否将这个variable添加到反向传播内
variable = Variable(tensor,requires_grad=True)
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

# d(v_out)/d(var) = 1/4 * 2 * variable = 1/2 * variable
v_out.backward()
print(variable.grad,"\n",variable.data)
