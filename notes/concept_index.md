# CS234 Concept Index

这个索引记录概念在课程笔记中**第一次被完整讲解**的位置。后续 lecture 再次使用同一概念时，应引用这里的位置，只解释本次新增内容；如果学习者要求重新讲解，仍保留原始位置并明确标注“重新完整讲解”。

## 使用规则

- `首次完整讲解` 指包含准确定义、白话解释和必要具体例子的首次教学位置，不等同于课件第一次提到。
- 新概念只在笔记小节编号稳定后登记。
- Aliases 用于识别中文、英文、缩写和常用数学符号。
- 这个索引记录内容覆盖，不表示学习者已经掌握。

## Registry

| ID | 概念 | Aliases | 首次完整讲解 | 备注 |
|---|---|---|---|---|
| sequential-decision-process | 序列决策过程（Sequential Decision Process） | agent-environment loop, time step | Lecture 1 §6 | 离散时间交互 |
| history | 历史（History） | $h_t$ | Lecture 1 §7.1 | 课件采用动作—观测—奖励序列 |
| state | 状态（State） | $s_t$, information state | Lecture 1 §7.2 | 历史的预测性摘要 |
| observation | 观测（Observation） | $o_t$ | Lecture 1 §7.3 | 不一定是 Markov state |
| markov-assumption | Markov 假设（Markov Assumption） | Markov property | Lecture 1 §8 | 当前状态是历史的充分统计量 |
| transition-model | 转移模型（Transition Model） | dynamics, $P(s'\mid s,a)$ | Lecture 1 §10 | Named environment 必须对照课件或代码 |
| reward-model | 奖励模型（Reward Model） | $R(s,a)$, expected reward | Lecture 1 §10 | 与实际奖励随机变量区分 |
| policy | 策略（Policy） | $\pi$, deterministic policy, stochastic policy | Lecture 1 §11 | §11.1–11.2 分别讲两类策略 |
| evaluation-control | 评估与控制（Evaluation vs Control） | prediction, control | Lecture 1 §12 | 给定策略估值 vs 寻找最优策略 |
| markov-process | Markov 过程（Markov Process） | Markov chain, MP | Lecture 1 §13 | 无动作、无奖励 |
| mrp | Markov 奖励过程（MRP） | Markov reward process | Lecture 1 §14 | Markov chain + rewards + discount |
| horizon | 时域长度（Horizon） | $H$, episode length | Lecture 1 §15.1 | 区分总步数与剩余步数 |
| return | 回报（Return） | $G_t$, discounted return | Lecture 1 §15.2 | 默认总 episode 长度约定 |
| state-value | 状态价值函数（State-value Function） | $V(s)$, $V_t(s)$, $V^\pi(s)$ | Lecture 1 §15.3 | 有限时域价值需带时间或剩余步数 |
| discount-factor | 折扣因子（Discount Factor） | $\gamma$ | Lecture 1 §16 | 控制远期奖励权重 |
| mrp-bellman-equation | MRP Bellman 方程 | Bellman expectation equation | Lecture 1 §17 | 一步奖励 + 下一状态价值期望 |
| matrix-bellman-equation | 矩阵 Bellman 方程 | $V=R+\gamma PV$ | Lecture 1 §18 | 有限状态 MRP |
| mrp-analytic-solution | MRP 解析解 | $(I-\gamma P)^{-1}R$ | Lecture 1 §19 | 实现时优先解线性方程而非显式求逆 |
| iterative-dp | 迭代动态规划（Iterative DP） | iterative policy evaluation precursor | Lecture 1 §20 | Bellman backup 反复传播价值 |
| agent-components | RL Agent 组件 | model, policy, value function | Lecture 1 §21 | model-based 与 model-free |
| policy-value | 策略价值函数（Policy Value） | $V^\pi(s)$ | Lecture 1 §22 | 价值依赖所执行策略 |
| mdp | Markov 决策过程（MDP） | $\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)$ | Lecture 2 §4 | MRP + actions |
| policy-induced-mrp | 策略诱导的 MRP | $R^\pi$, $P^\pi$ | Lecture 2 §6 | 对动作按策略概率边缘化 |
| policy-evaluation | 策略评估（Policy Evaluation） | Bellman expectation backup | Lecture 2 §7 | 给定策略计算 $V^\pi$ |
| optimal-value-policy | 最优价值与最优策略 | $V^*$, $\pi^*$, MDP control | Lecture 2 §11 | 最优价值唯一，策略可不唯一 |
| policy-iteration | 策略迭代（Policy Iteration） | PI | Lecture 2 §12 | evaluation + improvement |
| action-value | 状态—动作价值函数（Action-value Function） | $Q^\pi(s,a)$, Q-value | Lecture 2 §13 | 首动作固定，之后按 $\pi$ |
| policy-improvement | 策略改进（Policy Improvement） | greedy improvement | Lecture 2 §14 | 对 $Q^{\pi_i}$ 取 argmax |
| bellman-optimality-operator | Bellman 最优算子 | $B$, optimality backup | Lecture 2 §17 | 对动作取最大值 |
| value-iteration | 值迭代（Value Iteration） | VI, $V_{k+1}=BV_k$ | Lecture 2 §18 | 反复应用最优 Bellman backup |
| policy-bellman-operator | 固定策略 Bellman 算子 | $B^\pi$ | Lecture 2 §20 | fixed point 为 $V^\pi$ |
| contraction | 收缩映射（Contraction） | $\gamma$-contraction, infinity norm | Lecture 2 §21 | VI 收敛基础 |
| finite-horizon-vi | 有限时域值迭代 | $V_k$, $\pi_k$ | Lecture 2 §23 | 策略通常依赖剩余步数 |
| simulation-evaluation | 仿真策略评估（Simulation Evaluation） | rollout average | Lecture 2 §24 | 对完整 return 求样本平均 |
| bootstrapping | 自举（Bootstrapping） | bootstrap target | Lecture 3 §3.1 | 用当前估计更新估计 |
| model-free-policy-evaluation | 无模型策略评估 | direct experience | Lecture 3 §3 | 不读取真实 $P,R$ |
| monte-carlo-evaluation | 蒙特卡洛策略评估 | MC prediction | Lecture 3 §4 | 完整 episode return 平均 |
| first-visit-mc | 首次访问 MC | first-visit Monte Carlo | Lecture 3 §4.1 | 每条 episode 每状态至多一个样本 |
| every-visit-mc | 每次访问 MC | every-visit Monte Carlo | Lecture 3 §4.2 | 使用每次状态访问的 return |
| incremental-mc | 增量 MC | sample-average update | Lecture 3 §4.4 | 不保存全部历史回报 |
| learning-rate | 学习率（Learning Rate） | $\alpha$, step size | Lecture 3 §4.4 | target-error 更新骨架 |
| bias-variance-mse | 偏差、方差与 MSE | bias, variance, mean squared error | Lecture 3 §5.2 | MSE = variance + bias squared |
| consistency | 一致性（Consistency） | consistent estimator | Lecture 3 §5.3 | 与无偏性不同 |
| stochastic-approximation | 随机逼近学习率条件 | Robbins–Monro conditions | Lecture 3 §5.5 | 两个无穷级数条件 |
| td-learning | 时序差分学习（TD Learning） | TD(0), temporal difference | Lecture 3 §4.5 | 采样 + bootstrap；§6 算法化展开 |
| td-target | TD 目标 | $r_t+\gamma V(s_{t+1})$ | Lecture 3 §4.5 | 一步真实奖励 + 下一状态估计；§6.2 固定符号 |
| td-error | TD 误差与更新 | $\delta_t$, TD update | Lecture 3 §4.5 | target − current prediction；§6.2 固定符号 |
| certainty-equivalence | 确定性等价（Certainty Equivalence） | empirical MDP, MLE model | Lecture 3 §8 | 先估计模型，再动态规划 |
| batch-policy-evaluation | 批量策略评估 | batch MC, batch TD | Lecture 3 §9 | 固定有限数据反复更新 |
| model-free-control | 无模型控制（Model-free Control） | model-free policy iteration, control without a model | Lecture 4 §3.1 | 从经验估计 Q 并改进策略 |
| exploration-exploitation | 探索—利用权衡 | exploration, exploitation | Lecture 4 §3.1 | 探索获得信息，利用获得当前回报 |
| epsilon-greedy | epsilon-greedy 策略 | $\varepsilon$-greedy, epsilon policy | Lecture 4 §3.2 | 贪心动作加均匀随机探索 |
| epsilon-greedy-improvement | epsilon-greedy 单调改进 | monotonic epsilon-greedy policy improvement | Lecture 4 §3.3 | 相对同一个 $Q^{\pi_i}$ 的价值不下降 |
| monte-carlo-control | 蒙特卡洛控制 | Monte Carlo control, MC control | Lecture 4 §3.4 | 对 $Q(s,a)$ 使用完整 return |
| online-mc-control | 在线蒙特卡洛控制 | online MC control, on-policy improvement | Lecture 4 §3.4 | 评估、探索、改进交替进行 |
| glie | 无限探索极限贪心 | GLIE, greedy in the limit with infinite exploration | Lecture 4 §3.5 | 每个状态动作对无限访问且策略极限贪心 |
| on-policy-off-policy | 同策略与异策略 | on-policy, off-policy, behavior policy, target policy | Lecture 4 §4.1 | 数据策略与目标策略的关系 |
| td-control | 时序差分控制 | TD control, model-free policy iteration with TD | Lecture 4 §4.2 | 用 TD 估计动作价值并改进策略 |
| q-learning | Q-learning | off-policy TD control, $Q^*$ update | Lecture 4 §4.3 | target 对下一动作取最大值 |
| sarsa | SARSA | on-policy TD control | Lecture 4 §4.4 | target 使用实际下一动作 |
| function-approximation | 函数逼近 | value function approximation, VFA | Lecture 4 §5.1 | 用共享参数表示价值函数 |
| oracle-value-approximation | oracle 下的价值函数逼近 | supervised value approximation, oracle regression | Lecture 4 §5.2 | 用真实 $Q^\pi$ 标签构造监督学习目标 |
| stochastic-gradient-descent | 随机梯度下降 | SGD, stochastic gradient | Lecture 4 §5.2 | 用样本梯度更新参数向量 |
| mc-value-function-approximation | MC 函数逼近评估 | Monte Carlo VFA, MC value approximation | Lecture 4 §5.3 | 用完整 return 代替真实动作价值标签 |
| semi-gradient-td | 半梯度 TD | semi-gradient, TD with approximation | Lecture 4 §5.4 | target 依赖估计但更新时不对 target 反传 |
| vfa-control | 函数逼近控制 | control with value function approximation | Lecture 4 §5.5 | 近似评估与 epsilon-greedy 改进交替 |
| deadly-triad | 致命三角 | deadly triad | Lecture 4 §5.6 | 函数逼近、bootstrap、off-policy 的不稳定组合 |
| vfa-projection | 函数逼近投影 | projection, fitting back to function class | Lecture 4 §5.6 | 把 backup 结果映射回可表示函数类 |
| dqn | 深度 Q 网络 | DQN, Deep Q-Network | Lecture 4 §6.1 | 用神经网络近似 Q-learning |
| experience-replay | 经验回放 | replay buffer, experience replay | Lecture 4 §6.2 | 打散相关性并重复利用 transition |
| target-network | 目标网络 | fixed Q-targets, $w^-$ | Lecture 4 §6.3 | 暂时冻结 target 参数 |
| minibatch | 小批量 | minibatch, batch size $B$ | Lecture 4 §6.4 | 一次参数更新使用的一组 transition |
| dqn-pseudocode | DQN 训练流程 | DQN pseudocode, target synchronization | Lecture 4 §6.5 | replay、target、minibatch 和同步周期 |
| convolutional-network | 卷积神经网络 | CNN, convolutional neural network | Lecture 4 §6.6 | 从像素状态提取共享特征 |
| ablation-study | 消融实验 | ablation, component ablation | Lecture 4 §6.7 | 固定任务并改变组件以比较贡献 |
| policy-based-rl | 基于策略的强化学习 | policy-based RL, direct policy search | Lecture 5 §2.2 | 直接参数化动作分布 |
| state-aliasing | 状态混叠 | state aliasing, aliased states, perceptual aliasing | Lecture 5 §2.3 | 多个真实状态共享同一表示 |
| policy-gradient | 策略梯度 | policy gradient, $\nabla_\theta J(\theta)$ | Lecture 5 §2.4 | 对策略参数做梯度上升 |
| trajectory-distribution | 轨迹分布 | trajectory probability, $P(\tau;\theta)$ | Lecture 5 §3.1 | 初始分布、策略与 dynamics 的乘积 |
| likelihood-ratio-gradient | 似然比梯度恒等式 | likelihood-ratio trick, log-derivative trick | Lecture 5 §3.2 | $\nabla p=p\nabla\log p$ |
| score-function | 得分函数 | score function, $\nabla_\theta\log p(x;\theta)$ | Lecture 5 §3.2 | 提高样本 log-probability 的参数方向 |
| softmax-policy | Softmax 策略 | categorical policy, Gibbs policy | Lecture 5 §4.1 | 离散动作概率参数化 |
| gaussian-policy | Gaussian 策略 | normal policy, continuous-action policy | Lecture 5 §4.2 | 连续动作分布参数化 |
| policy-gradient-theorem | 策略梯度定理 | policy gradient theorem | Lecture 5 §4.3 | score 与 $Q^\pi$ 的期望 |
| reinforce | REINFORCE | Monte Carlo policy gradient | Lecture 5 §5.2 | 用 sampled reward-to-go 估计 $Q^\pi$ |
| policy-gradient-baseline | Policy gradient baseline | state baseline, control variate, $b(s)$ | Lecture 5 §5.4 | action-independent baseline 不改变期望梯度 |
| variance-optimal-baseline | 单个梯度项的方差最优 baseline | optimal control variate, score-norm-weighted baseline | Lecture 6 §2.2 | 固定状态的单项最优解；$V^\pi(s)$ 是常用近似 |
| vanilla-policy-gradient | Vanilla policy gradient | VPG | Lecture 6 §2.3 | rollout、return、baseline、policy update 循环 |
| advantage-function | 优势函数 | advantage, $A^\pi(s,a)$ | Lecture 6 §2.4 | $Q^\pi(s,a)-V^\pi(s)$ |
| actor-critic | Actor-critic | actor, critic | Lecture 6 §2.4 | 显式策略与价值估计共同更新 |
| n-step-target | N-step target | multi-step return, $\hat Q_t^{(n)}$ | Lecture 6 §2.5 | 来源补充；真实奖励与 value bootstrap 的混合 |
| on-policy-data-reuse | On-policy 数据复用限制 | policy-gradient sample efficiency, stale policy data | Lecture 6 §3.1 | 更新后旧 batch 与 current policy 出现分布 mismatch |
| policy-space-distance | 策略空间距离 | parameter distance vs policy distance | Lecture 6 §3.3 | 小参数步不保证小分布变化 |
| discounted-state-distribution | 折扣状态分布 | discounted visitation distribution, $d^\pi$ | Lecture 6 §4.1 | 跨时间步的 normalized discounted mixture |
| performance-difference-lemma | 策略性能差异引理 | performance difference lemma, relative policy performance | Lecture 6 §4.2 | 用旧策略 advantage 表达新旧真实性能差 |
| policy-probability-ratio | 策略概率比 | importance ratio, $\pi'/\pi$, $r_t(\theta)$ | Lecture 6 §4.3 | 单步 ratio 只修正动作分布 mismatch |
| policy-surrogate-objective | 策略代理目标 | surrogate objective, $\mathcal L_\pi(\pi')$ | Lecture 6 §4.4 | 以 $d^{\pi'}\approx d^\pi$ 换取旧数据估计 |
| kl-divergence | KL 散度 | Kullback--Leibler divergence, relative entropy | Lecture 6 §4.5 | 非负但不对称的 policy-space 变化度量 |
| ppo | 近端策略优化 | PPO, Proximal Policy Optimization | Lecture 6 §5.1 | Adaptive KL penalty 与 clipped objective |
| ppo-clipped-objective | PPO clipped objective | clipped surrogate, $\mathcal L^{\mathrm{CLIP}}$ | Lecture 6 §5.2 | 按 advantage 符号截断有利方向的 ratio 激励 |
| n-step-advantage | N-step 优势估计 | N-step advantage estimator, $\hat A_t^{(k)}$ | Lecture 7 §2.2 | 真实奖励与 value bootstrap 的 $k$ 步组合 |
| gae | 广义优势估计 | Generalized Advantage Estimation, GAE, $\lambda$ | Lecture 7 §2.3 | 对 N-step estimator 指数加权；$\lambda$ 调整 bias--variance |
| monotonic-improvement-bound | 单调改进性能下界 | monotonic improvement theory, performance bound, trust region | Lecture 7 §3.1 | surrogate gain 减 KL policy-shift 代价 |
| imitation-learning | 模仿学习 | imitation learning, learning from demonstrations | Lecture 7 §4.1 | 从专家行为学习 policy 或 reward |
| behavioral-cloning | 行为克隆 | behavioral cloning, BC | Lecture 7 §4.2 | 在示范 $(s,a)$ 对上做监督学习 |
| compounding-errors | 误差累积 | compounding errors, distribution mismatch | Lecture 7 §4.3 | 学习策略诱导的状态分布偏离专家分布 |
| dagger | 数据集聚合 | DAGGER, Dataset Aggregation | Lecture 7 §4.4 | 在 learner-induced states 上查询专家并聚合标签 |
| inverse-rl | 逆强化学习 | inverse reinforcement learning, IRL | Lecture 7 §5.1 | 从专家示范反推可能的 reward |
| feature-reward | 线性特征奖励 | feature-based reward, $R(s)=w^\top x(s)$ | Lecture 7 §5.2 | reward 对 state features 线性组合 |
| discounted-feature-expectation | 折扣特征期望 | discounted feature expectations, $\mu^\pi$ | Lecture 7 §5.2 | 把 policy value 写成 $w^\top\mu^\pi$ |
| feature-matching | 特征匹配 | feature matching, apprenticeship learning | Lecture 7 §5.3 | feature expectation 接近可保证有界 reward gap |
| maxent-irl | 最大熵逆强化学习 | Maximum Entropy IRL, MaxEnt IRL | Lecture 7 §5.4 | 在匹配示范统计量外不添加额外路径偏好 |
| maxent-path-distribution | 最大熵路径分布 | exponential-family path distribution, $P(\tau\mid w)$ | Lecture 7 §5.4 | 按 $\exp(w^\top\mu_\tau)$ 对路径加权 |
| importance-sampling | 重要性采样 | importance sampling, $P/Q$ weight | Lecture 7 §6.1 | 用 proposal 样本估计 target expectation |
| off-policy-policy-gradient | 异策略策略梯度 | off-policy policy gradient, prefix importance ratio | Lecture 7 §6.2 | 轨迹 prefix ratio 连乘导致高方差 |
| human-preference-learning | 人类偏好学习 | human feedback, pairwise preference, preference learning | Lecture 8 §2.3 | 用比较反馈替代逐状态 action 或手写 reward |
| pairwise-comparison | 成对比较 | pairwise comparison, preference label | Lecture 8 §2.4 | 比较两个候选行为而非给绝对标量评分 |
| bradley-terry | Bradley--Terry 偏好模型 | Bradley--Terry model, paired comparison, $\sigma(r_i-r_j)$ | Lecture 8 §3.1 | 用 reward difference 定义偏好概率 |
| preference-reward-model | 偏好奖励模型 | preference reward model, reward model, $r_\phi$, trajectory score | Lecture 8 §3.2 | 从 pairwise labels 学习可优化的代理 reward |
| rlhf | 人类反馈强化学习 | reinforcement learning from human feedback, RLHF | Lecture 8 §4.2 | reward model + KL-regularized policy optimization |
| kl-regularized-policy | KL 正则策略目标 | KL-regularized RL, reference policy, $\beta$ | Lecture 8 §4.3 | 在 reward 与 reference policy proximity 间折中 |
| dpo | 直接偏好优化 | Direct Preference Optimization, DPO | Lecture 8 §5.2 | 由 reward-policy transformation 直接优化 policy |
| receding-horizon-control | 滚动时域控制 | receding-horizon control, model predictive control, MPC | Lecture 8 §5.5 | 预测 action segment，只执行第一步后重新规划 |
