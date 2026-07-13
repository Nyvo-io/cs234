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
| bootstrapping | 自举（Bootstrapping） | bootstrap target | Lecture 3 §5 | 用当前估计更新估计 |
| model-free-policy-evaluation | 无模型策略评估 | direct experience | Lecture 3 §6 | 不读取真实 $P,R$ |
| monte-carlo-evaluation | 蒙特卡洛策略评估 | MC prediction | Lecture 3 §7 | 完整 episode return 平均 |
| first-visit-mc | 首次访问 MC | first-visit Monte Carlo | Lecture 3 §8 | 每条 episode 每状态至多一个样本 |
| every-visit-mc | 每次访问 MC | every-visit Monte Carlo | Lecture 3 §9 | 使用每次状态访问的 return |
| incremental-mc | 增量 MC | sample-average update | Lecture 3 §11 | 不保存全部历史回报 |
| learning-rate | 学习率（Learning Rate） | $\alpha$, step size | Lecture 3 §12 | target-error 更新骨架 |
| bias-variance-mse | 偏差、方差与 MSE | bias, variance, mean squared error | Lecture 3 §15 | MSE = variance + bias squared |
| consistency | 一致性（Consistency） | consistent estimator | Lecture 3 §16 | 与无偏性不同 |
| stochastic-approximation | 随机逼近学习率条件 | Robbins–Monro conditions | Lecture 3 §18 | 两个无穷级数条件 |
| td-learning | 时序差分学习（TD Learning） | TD(0), temporal difference | Lecture 3 §20 | 采样 + bootstrap |
| td-target | TD 目标 | $r_t+\gamma V(s_{t+1})$ | Lecture 3 §21 | 一步真实奖励 + 下一状态估计 |
| td-error | TD 误差与更新 | $\delta_t$, TD update | Lecture 3 §22 | target − current prediction |
| certainty-equivalence | 确定性等价（Certainty Equivalence） | empirical MDP, MLE model | Lecture 3 §29 | 先估计模型，再动态规划 |
| batch-policy-evaluation | 批量策略评估 | batch MC, batch TD | Lecture 3 §31 | 固定有限数据反复更新 |
