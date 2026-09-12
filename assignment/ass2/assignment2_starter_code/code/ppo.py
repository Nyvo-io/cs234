import numpy as np
import torch
import torch.nn.functional as F
import gym
import itertools
import copy
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient

#这里ppo采用的是clipped objective，不是 KL penalty
#防止一次参数更新，把policy改的太猛
#所以引入 ratio
class PPO(PolicyGradient):

    def __init__(self, env, config, seed, logger=None):
        config.use_baseline = True
        super(PPO, self).__init__(env, config, seed, logger)
        self.eps_clip = self.config.eps_clip

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.

        To compute the loss value, you will need the log probabilities of the actions
        given the observations as before. Note that the policy's action_distribution
        method returns an instance of a subclass of torch.distributions.Distribution,
        and that object can be used to compute log probabilities.

        Note:
            - PyTorch optimizers will try to minimize the loss you compute, but you
            want to maximize the policy's performance.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        #πθold​​，旧策略
        old_logprobs = np2torch(old_logprobs)
        #-
        #observations这些是用旧策略采样得到的数据
        #我们利用刚才rollout里面采取过的动作
        #得到当前policy的策略分布
        distribution = self.policy.action_distribution(observations)
        #取log，logπθ​(at​∣st​)。当前的新 policy 对“刚才那个动作”认为概率是多少？
        new_logprobs = distribution.log_prob(actions)

        #如果是高斯分布，连续的动作空间，他的动作一般不是一个数，而是一个动作向量
        #at​=[at,1​, at,2​, at,3​] -> action = [0.2, -0.5, 0.7]
        #log_prob 会分别算每一维，假设batch 里有 4 个 observation，对应的action
        #[
        #[-0.3, -0.5, -0.2],
        #[-0.7, -0.1, -0.4],
        #[-0.2, -0.6, -0.3],
        #[-0.4, -0.2, -0.5]
        #]
        #new_logprobs.shape：这个tensor有几个维度
        #我们要计算的是整体的π(at​∣st​)，得把上面几个动作乘起来
        #π(at​∣st​)=π(at,1​∣st​)π(at,2​∣st​)π(at,3​∣st​)，转成log，就是相加
        if len(new_logprobs.shape) > 1:
            new_logprobs = torch.sum(new_logprobs, dim=-1)

        advantages = advantages.squeeze(-1)

        #ratio就是这个状态选这个动作的新旧概率比值
        ratio = torch.exp(new_logprobs - old_logprobs)

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(
            ratio,
            1 - self.eps_clip,
            1 + self.eps_clip
        ) * advantages

        loss = -torch.mean(torch.min(surrogate1, surrogate2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
        #######################################################
        #########   YOUR CODE HERE - 10-15 lines.   ###########

        #######################################################
        #########          END YOUR CODE.          ############

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
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, 
                                   old_logprobs)

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

    #用旧策略(初始策略)采样数据
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
            states, actions, old_logprobs, rewards = [], [], [], []
            #一整个episode的总reward
            episode_reward = 0
            #一个episode里最多有self.config.max_ep_len步
            for step in range(self.config.max_ep_len):
                #先保存state
                states.append(state)
                # Note the difference between this line and the corresponding line
                # in PolicyGradient.
                #policy.act：根据已有的策略，输入state采样输出action。同时计算他的log
                #states[-1][None]:-1指的是选取列表的最后一个元素，也就是state
                #None：增加一个batch维度。因为神经网络一般希望输入[batch_size, observation_dim]
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob = True)
                assert old_logprob.shape == (1,)

                action, old_logprob = action[0], old_logprob[0]
                #让环境执行该动作，看看到了那个新状态，得到的reward是多少
                #它是是 RL 最经典的 transition
                state, reward, done, info = env.step(action)
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break
            
            #一局结束，把整个trajectory 打包
            #path = {
            #   "observation": [s0, s1, s2, s3],
            #    "action":      [a0, a1, a2, a3],
            #    "reward":      [r0, r1, r2, r3],
            #    "old_logprobs":[lp0,lp1,lp2,lp3]
            #}
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards
