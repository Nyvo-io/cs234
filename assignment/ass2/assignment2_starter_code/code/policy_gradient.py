import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy

#batch_size是指总共有几个timestep
#假如paths里有5个path，第一个path有200个step，第二个有100个
#加起来如果有1000个path，那么这里的batch_size就是1000
#把所有都平铺开

#实现策略梯度算法
#策略梯度训练器
class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """
    #对PolicyGradient训练器 进行初始化
    #config是超参数，env是环境，seed是随机种子
    def __init__(self, env, config, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment,虚拟环境
                config: class with hyperparameters
                logger: logger instance from the logging module

        You do not need to implement anything in this function. However,
        you will need to use 
        self.discrete, self.observation_dim,
        self.action_dim, and self.lr in other methods.

        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.env.seed(self.seed)

        # discrete vs continuous action space
        #创建discreet 对象
        # 判断运动空间是离散还是连续，
        #利用instance()函数判断env.action_space是否是gym.spaces.Discrete的实例
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.lr = self.config.learning_rate

        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(env, config)

    #初始化策略
    def init_policy(self):
        """
        Please do the following:
        1. Create a network using build_mlp. It should map vectors of size
           self.observation_dim to vectors of size self.action_dim, and use
           the number of layers and layer size from self.config
        2. If self.discrete = True (meaning that the actions are discrete, i.e.
           from the set {0, 1, ..., N-1} where N is the number of actions),
           instantiate a CategoricalPolicy.
           If self.discrete = False (meaning that the actions are continuous,
           i.e. elements of R^d where d is the dimension), instantiate a
           GaussianPolicy. Either way, assign the policy to self.policy
        3. Create an Adam optimizer for the policy, with learning rate self.lr
           Note that the policy is an instance of (a subclass of) nn.Module, so
           you can call the parameters() method to get its parameters.
        """
        #创建神经网络（MLP）
        #根据动作空间选择 CategoricalPolicy 或 GaussianPolicy
        #创建 optimizer

        #创建神经网络
        self.network = build_mlp(
            input_size=self.observation_dim,
            output_size=self.action_dim,
            n_layers=self.config.n_layers,
            size=self.config.layer_size,
        )

        #根据当前动作空间选择policy，是离散型还是连续型高斯分布
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim)
        
        
        #创建optimizer
        #adam是优化算法
        #根据梯度更新网络参数
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr
        )
        #######################################################
        #########   YOUR CODE HERE - 8-12 lines.   ############

        #######################################################
        #########          END YOUR CODE.          ############


    #属于训练过程中的统计记录：logging
    #记录训练过程中 reward 的变化
    #方便画曲线
    #判断训练有没有变好
    def init_averages(self):
        """
        You don't have to change or use anything here.
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0
    #每训练一轮，更新统计量
    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    #收集训练数据
    #使用当前策略在环境中跑若干个 episode，采集多条轨迹，都存入paths
    #然后之后会把多条轨迹组合成一个batch，也就是一整条轨迹，一起投入训练
    #不然单条轨迹取期望，方差太大
    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.policy.act(states[-1][None])[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    

    #已经完成πθ​(a∣s)，并且采集到了(st,at,rt)
    #现在需要用r，计算每一条轨迹的G，从后往前算
    #这个Gt计算出来，之后用来进行策略更新
    #用Gt ​≈ Qπ(st​,at​)
    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path

        TODO: compute and return G_t for each timestep. Use self.config.gamma.
        """

        all_returns = []
        #读取每一条轨迹
        for path in paths:
            rewards = path["reward"]

            #保存[G0​,G1​,G2​,...]，一条轨迹的每个时间步的G
            returns = []
            #最后一步之后：GT+1​=0
            G = 0

            for r in reversed(rewards):
                G = r + self.config.gamma * G
                #list.insert(index, value)：在index位置插入元素
                returns.insert(0,G)
            #######################################################
            #########   YOUR CODE HERE - 5-10 lines.   ############

            #######################################################
            #########          END YOUR CODE.          ############

            all_returns.append(returns)
            #all_returns存paths里面所以轨迹的returns，是二维列表
        #concatenate可以把他们前后拼接在一起，形成一个batch，方便后续训练策略取值
        returns = np.concatenate(all_returns)

        return returns

    #归一化是对整个batch的advantage做
    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]

        TODO:
        Normalize the advantages so that they have a mean of 0 and standard
        deviation of 1. Put the result in a variable called
        normalized_advantages (which will be returned).

        Note:
        This function is called only if self.config.normalize_advantage is True.
        """

        #policy gradient: ∇θ​J=E[∇θ​logπθ​(at​∣st​)At​]
        #如果advantage 很大，梯度会比较大，训练会不稳定，所以进行标准化
        #这里是一个batch的多条轨迹的advantage，他们形成一个数组，不再是单个值
        normalized_advantages = (
            (advantages - np.mean(advantages)) / np.std(advantages)
        )
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############
        return normalized_advantages
   
    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        #使用baseline的话就是用At，不使用baseline的话At就等于Gt
        #config里有要求采用baseline方法降低方差的话
        #就采用baseline_network里的方式计算A，init里有初始化
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        #如果配置打开，就进行标准化操作
        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages



    #根据数据更新策略
    #θ ← θ+α∇θ​J(θ)
    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data.
        To compute the loss, you will need the log probabilities of the actions
        given the observations. Note that the policy's action_distribution
        method returns an instance of a subclass of
        torch.distributions.Distribution, and that object can be used to
        compute log probabilities.
        See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
        PyTorch optimizers will try to minimize the loss you compute, but you
        want to maximize the policy's performance.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)

        #根据动作空间是连续还是离散，输出对应的 π(·|s)
        distribution = self.policy.action_distribution(observations)

        #输入batch里的动作序列，然后根据对应策略，输出并转成log形式
        log_probs = distribution.log_prob(actions)

        #构造loss，也就是把J取负。这个乘法是tensor运算
        #logπθ​(at​∣st​) * At​
        loss = -(log_probs * advantages).mean()
        #清除之前保留的梯度
        self.optimizer.zero_grad()
        #对参数求导
        loss.backward()
        #优化器默认做最小化
        self.optimizer.step()
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############


    #安排训练流程
    #train loop
    #for t in range(self.config.num_batches)，更新几次策略
    #当前策略 -> 采集轨迹 -> 计算G -> 计算A -> 更新神经网络参数 -> 新策略
    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()


#3:接下来写baseline_network.py，因为上面PolicyGradient.calculate_advantage()会调用这个类
#它目的是建立价值网络V(s)，从而得到A