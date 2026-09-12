import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch

#建立V(s) 价值神经网络
#根据状态预测这个状态未来平均能获得多少奖励
#BaselineNetwork类，继承了nn.Module
class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, etc.
        The output of the network has dimension 1.
        """
        super().__init__()
        self.config = config
        self.env = env
        #创建baseline属性
        #它就是负责计算V的 MLP
        self.baseline = None

        #学习率
        self.lr = self.config.learning_rate
        #observation 就是 st，用一个向量表示
        #[0]就是输出这个状态向量的维度，有几个元素表示这个向量，作为input size
        observation_dim = self.env.observation_space.shape[0]

        self.baseline = build_mlp(
            input_size=observation_dim,
            output_size=1,
            n_layers=self.config.n_layers,
            #每个隐藏层有多少个神经元
            size = self.config.layer_size
        )

        #优化器，用于训练神经网络
        self.optimizer = torch.optim.Adam(
            self.baseline.parameters(),
            lr=self.lr
        )
        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############

        #######################################################
        #########          END YOUR CODE.          ############

    #st​→ MLP →Vϕ​(st​)
    #forward 方法，调用该方法时，只用传入obs，selfBaselineNetwork对象自动构造
    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #对batch里的所有状态计算V，batch_size是指总共有几个timestep
        #假如paths里有5个path，第一个path有200个step，第二个有100个
        #加起来如果有1000个path，那么这里的batch_size就是1000
        #把所有都平铺开
        #[ [V(s0)],[V(s1)],[V(s2)],[V(s3)],[V(s4)] ]，他的shape是[5, 1]
        #我们要输出[V(s0), V(s1), V(s2), V(s3), V(s4)]，所以把shape最后一维取消，变成-1
        output = self.baseline(observations).squeeze(-1)
        #######################################################
        #########   YOUR CODE HERE - 1 lines.     #############

        #######################################################
        #########          END YOUR CODE.          ############
        assert output.ndim == 1
        return output

    #之前计算出了V，现在用Q减去它，计算开启baseline模式后的Advantage
    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)

        #这里的self指BaselineNetwork，
        #这里的self(obs)会调用self.forward，这是在nn.module里定好的
        #但module里只定好了调用流程，具体实现会调用你重写的forward
        #self():返回[V(s0), V(s1), V(s2), V(s3), V(s4)]
        #detach().cpu().numpy()：把tensor转换成数组
        #detach：不用再通过它反向传播梯度，计算图，训练网络
        #再移到cpu上
        baseline = self(observations).detach().cpu().numpy()
        advantages = returns - baseline
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############

        #######################################################
        #########          END YOUR CODE.          ############
        return advantages

    #专门训练baseline神经网络
    #上面的detach就是把这两个训练过程分开，上面是使用
    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer once.

        You may find the following documentation useful:
        https://pytorch.org/docs/stable/nn.functional.html
        """
        returns = np2torch(returns)
        observations = np2torch(observations)

        predictions = self(observations)
        loss = torch.nn.functional.mse_loss(predictions,returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #######################################################
        #########   YOUR CODE HERE - 4-10 lines.  #############

        #######################################################
        #########          END YOUR CODE.          ############


#他已经可以成功接入PolicyGradient(use_baseline=True)
#再写ppo