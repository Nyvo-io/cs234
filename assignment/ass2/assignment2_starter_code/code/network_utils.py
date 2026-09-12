import numpy as np
import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    #输入维度 输出维度 隐藏层层数 每个隐藏层 neuron 数
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    It should have 'n_layers' layers, each of 'size' units and followed
    by a ReLU nonlinearity. Additionally, the final layer should be linear (no ReLU).

    That is, the network architecture should be the following:
    [LINEAR LAYER]_1 -> [RELU] -> [LINEAR LAYER]_2 -> ... -> [LINEAR LAYER]_n -> [RELU] -> [LINEAR LAYER]

    "nn.Linear" and "nn.Sequential" may be helpful.
    """
    layers = []
    #使得这么一层层的叠加起来，写网络常见写法
    #[
    #Linear(4,64),
    #ReLU(),
    #Linear(64,64)
    # ReLU(),]

    #first hidden layer
    layers.append(nn.Linear(input_size, size))
    layers.append(nn.ReLU())

    #remaining hidden layers
    #只需要控制循环次数，不关心循环变量i，所以用_
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(nn.ReLU())

    #output layer
    layers.append(nn.Linear(size, output_size))

    return nn.Sequential(*layers)
    #nn.Sequential()：把多个神经网络层按照顺序连接起来
    #如果没有*，则会把layers当作一个整体参数传入nn.Sequential()，而不是把layers中的每个元素作为单独的参数传入
    #列表只是存东西,是一个普通容器,而nn.Sequential 是一个真正的神经网络模型。所以得先把列表东西取出
    #一层层存入Sequential
    #一个 PyTorch 模型，本质需要：model(x)

    #######################################################
    #########   YOUR CODE HERE - 7-15 lines.   ############

    #######################################################
    #########          END YOUR CODE.          ############
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
