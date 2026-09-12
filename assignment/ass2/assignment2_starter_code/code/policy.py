import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


#定义policy的基类，所有policy都继承这个类
#执行
#action_distribution()
#act()

#batch_size是指总共有几个timestep
#假如paths里有5个path，第一个path有200个step，第二个有100个
#加起来如果有1000个path，那么这里的batch_size就是1000
#把所有都平铺开

class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        这个函数的目的是输入状态s,返回一个分布对象，表示在这个状态下采取动作的概率分布：π(·|s)
        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        #需要子类重写，因为Base policy不知道你的动作空间
        #是连续的还是离散的
        raise NotImplementedError 

    #得出具体动作
    #环境给obs
    #比如 action=policy.act(obs) -> 输出左/右
    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution.
        -> 调用self.action_distribution()获取动作分布π(·|s); 所有动作的概率集合，然后从该分布中采样。
        Compute the log probability of the sampled actions using self.action_distribution. 
        -> 采样动作的概率 然后去对数,logπ(a|s),方便用于计算梯度,PPO
        You will have to convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        #把 numpy 转成 pytorch tensor
        # 因为环境都是处理numpy array：obs=np.array([0.1,0.2,0.3])
        #但要输入神经网络就得转换成tensor
        #但神经网络(nn.Linear)需要 pytorch tensor：torch.tensor([[0.1,0.2,0.3]])
        observations = np2torch(observations)

        distribution = self.action_distribution(observations)
        #此时根据动作空间是离散还是连续，action_distribution已经被后续对象重写
        #distribution 获取动作分布π(·|s)

        #sample 按照动作概率分布随机采样一个动作
        sampled_actions = distribution.sample()

        #对采样的动作计算log probability
        log_probs = distribution.log_prob(sampled_actions)

        #再把 log_probs和action 的pytorch tensor 转成 numpy array
        #未转换前 比如：sampled_actions.shape = torch.Size([3, 4])，表示batch里有3个状态，每个状态4维动作
        #一个batch里有3个机器人状态
        #tensor([
        #    [0.52, 0.18, -0.05, 0.91],
        #    [0.06, 0.72,  0.20, 0.31],
        #    [-0.25,0.35, 0.90, 0.05]
        #])
      # 第一行对应s1的采样动作：a1​=[0.52,0.18,−0.05,0.91]

    #把它转换成array：
    #array([
    #[0.52, 0.18, -0.05, 0.91],
    #[0.06, 0.72, 0.20, 0.31],
    #[-0.25,0.35, 0.90,0.05]
    #])

    #神经网络里用tensor，但环境通常用numpy

        sampled_actions = sampled_actions.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()

        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############

        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


#离散动作策略：softmax
#继承了 BasePolicy 和 nn.Module 两个类
#self 就是 CategoricalPolicy 的实例，policy
#调用这个类，从init看出 就只需要传入一个network：policy = CategoricalPolicy(network)
class CategoricalPolicy(BasePolicy, nn.Module):

    #外面传入一个神经网络，保存self.network
    #self.network(observations)：状态 → 网络 → logits
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    #重写BasePolicy里的action_distribution函数，用于离散状态，返回πθ​(⋅∣s)
    #一般这个函数通过父类函数里的act()调用
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        batch里有3个状态,每个状态4维。那么 observations.shape = [3,4]

        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.o  rg/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############

        #输入状态s，经过网络fθ​(s)，输出：[2.0,1.0]
        logits = self.network(observations)

        #利用loggits创建一个Categorical分布对象，表示在这个状态下采取动作的概率分布：π(·|s)
        #pytorch 在这里 采用 softmax
        distribution = ptd.Categorical(logits=logits)

        #返回distribution，也就是π(·|s)
        return distribution


#和前面CategoricalPolicy 是同一个思想，只不过这里处理连续空间
class GaussianPolicy(BasePolicy, nn.Module):
    #action_dim 表示动作空间的维度
    #一次动作里需要同时输出多少个数
    #机器人机械臂 有6个关节：
    #a=[a1,a2,a3,a4,a5,a6]，所以action_dim=6
    #可以表示关节1角速度，关节2角速度...
    #每个关节速度都是一个高斯分布
    #最后gaussian 输出的也是μθ​(s)=[μ1,μ2,μ3,μ4,μ5,μ6]

    #离散动作类别数，是表示在神经网络的输出维度里了
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        #这里的标准差是可训练参数
        nn.Module.__init__(self)
        self.network = network

        # σ 是不经过神经网络的可训练参数，和W b不一样。要优化 θ={W,b,logσ}
        # 当前策略输出太随机，经常做坏动作，梯度会让σ变小
        # 均值由普通神经网络 Wx+b 学习，标准差作为额外的可训练参数，通过 PPO 的梯度一起更新
        # 训练过程中梯度更新σ 会小于0，但高斯分布要求标准差必须为正数，所以我们学习存放的是logσ
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))
        #创建一个全0 tensor，并且告诉pytorch，这是模型参数，需要训练

        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############

        #######################################################
        #########          END YOUR CODE.          ############

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]:返回的是动作空间维度数的tensor

        The return value contains the standard deviations(标准差) for each dimension
        of the policy's actions. It can be computed from self.log_std
        """

        std = torch.exp(self.log_std)
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############

        #######################################################
        #########          END YOUR CODE.          ############
        return std

    #返回πθ​(a∣s)，也就是 N(μθ​(s),σ)
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()
                #均值是神经网络生成的,标准差是self.std()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """

        mean = self.network(observations)
        
        #普通一维高斯：就只有一个动作，a1​∼N(μ1​,σ1​)
        #多元高斯：a=[a1​,a2​,a3​,a4​,a5​,a6​]，输出N(μ,Σ)
        #前者[μ1​,μ2​,...,μ6​]，后者叫协方差矩阵，不仅包含每个动作自己的随机程度
        #还包含不同动作的相关性，比如a1和a2，会相互影响
        #但本例，机器人两个关节a1和a2是独立的，所以要求协方差矩阵是对角矩阵，只有对角线有值，其他都是0
        #也就是diagonal Gaussian
        #但pytorch没有diagonal Gaussian，所以我们要自己造一个

        #方法1：用MultivariateNormal
       
        #方法2：用Normal + Independent，两个pytorch组合
        #Normal 是 PyTorch 里的：N(μ,σ)，也就是一维正态分布
        normal = ptd.Normal(mean, self.std()) #会自动创建action_dim个一维高斯分布

        distribution = ptd.Independent(normal, 1)
        #把action_dim个一维高斯分布组合成一个多元高斯，把他们组合成一个整体，输出一个分布对象

        #######################################################
        #########   YOUR CODE HERE - 2-4 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############
        return distribution


#CategoricalPolicy（离散动作）， 经过mlp， 输出loggits，经过softmax，得到动作概率分布π(·|s)
#GaussianPolicy（连续动作），经过mlp，输出μθ​(s)
#比如他的obs shape是[batch_size,6]，通过神经网络训练得到μ=[0.2,0.5,−0.1,0.8,0.3,0.4]
#再结合可学习的标准差  构造 Gaussian 分布