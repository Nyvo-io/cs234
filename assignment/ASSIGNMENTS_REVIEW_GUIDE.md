---
title: CS234 Ass2-Ass3 复习索引
tags:
  - cs234
  - assignment
  - review
aliases:
  - CS234 作业复习索引
---

# CS234 Ass2 / Ass3 复习索引

这份笔记用于开始代码或写题前定位需要复习的知识。建议每次只复习当前任务所需的小节，并在完成推导、代码检查或实验后勾选对应项目。

## 总体顺序

1. Ass2：Lecture 4 → Lecture 5 → Lecture 6 → Lecture 7 的 PPO/GAE 部分。
2. Ass2 先完成 DQN 和 REINFORCE，再做 baseline，最后做 PPO 和实验。
3. Ass3：先复习 Lecture 7 的模仿学习/IRL，再复习 Lecture 8 的 preference、RLHF、DPO。
4. Ass3 先做 Hopper reward engineering 和 PPO 基线，再做 reward model/RLHF，最后做 DPO。

---

## 快速入口：按提交部分复习

下面四张表是实际做作业时的导航；后面的章节提供公式、代码文件和实验检查细节。

### Ass2：书面部分

| 作业内容 | 先掌握的知识 | Lecture notes 位置 |
|---|---|---|
| §1.1(a) DQN 改写成 tabular Q-learning | tabular Q-learning、Bellman target、epsilon-greedy | [[../notes/lec_notes/lec4_notes#3. Tabular Monte Carlo Control]]、[[../notes/lec_notes/lec4_notes#4. Tabular TD Control：从 sample backup 到 Q-learning]] |
| §1.1(b) Mars Rover 变差的设置 | state aliasing、函数逼近动机、表格方法的状态空间限制 | [[../notes/lec_notes/lec3_notes]]、[[../notes/lec_notes/lec4_notes#6. Deep Q-Network]] |
| §1.1(c) replay buffer | 样本相关性、experience replay、target network | [[../notes/lec_notes/lec4_notes#6. Deep Q-Network]] |
| §2.7(a) $O(T)$ return | reward-to-go、反向递推 | [[../notes/lec_notes/lec5_notes#5.1 利用因果结构：从整条return到 reward-to-go]] |
| §2.7(b) PPO 零梯度 | clipped surrogate、正负 advantage 的分段行为 | [[../notes/lec_notes/lec6_notes#5.2 Clipped surrogate objective]] |
| §2.7(c) log-probability 缓存 | on-policy、old policy、importance ratio | [[../notes/lec_notes/lec6_notes#5.4 Assignment 2 中的 `PPO.update_policy`]] |
| 第 3 题分布证明 | trajectory distribution、discounted state distribution、performance difference lemma | [[../notes/lec_notes/lec6_notes#4. 从性能差异恒等式到可优化 surrogate]] |
| 第 4 题伦理 | respect for persons、beneficence、justice、IRB | 题面指定的 Belmont Report 与 Stanford IRB 材料；不属于 Lecture 6 的算法内容 |

### Ass2：代码部分

| 实现顺序 | 需要掌握的知识 | Lecture notes 位置 | 代码位置 |
|---|---|---|---|
| 1. MLP | `nn.Linear`、ReLU、Sequential、输入/输出 shape | [[../notes/lec_notes/lec5_notes#Assignment Readiness：Assignment 2]] | `ass2/assignment2_starter_code/code/network_utils.py` |
| 2. Policy distributions | categorical、Gaussian、sampling、联合 log-probability | [[../notes/lec_notes/lec5_notes#4. 策略参数化与 policy gradient theorem]] | `code/policy.py` |
| 3. REINFORCE | policy gradient theorem、return、flattened batch objective | [[../notes/lec_notes/lec5_notes#5. REINFORCE 与方差降低]] | `code/policy_gradient.py` |
| 4. Baseline | $V(s)$、advantage、MSE、normalization | [[../notes/lec_notes/lec6_notes#2.3 Vanilla policy gradient：把 policy 与 baseline 接成训练循环]] | `code/baseline_network.py` |
| 5. PPO | clipped ratio、old log-probability、batch reuse | [[../notes/lec_notes/lec6_notes#5. Proximal Policy Optimization]] | `code/ppo.py` |
| 6. 实验和 plot | seed、平均回报、方差、训练曲线 | [[../notes/lec_notes/lec6_notes#6. Assignment Readiness：Assignment 2]] | `code/main.py`、`code/plot.py` |

### Ass3：书面部分

| 作业内容 | 先掌握的知识 | Lecture notes 位置 |
|---|---|---|
| §1 reward engineering | reward misspecification、reward hacking、termination trade-off | [[../notes/lec_notes/lec7_notes#7. Assignment Readiness]]；Hopper 具体奖励项以题面/环境文档为准 |
| §2.1 reward-model gradient | Bradley–Terry、sigmoid、trajectory reward difference、交叉熵梯度 | [[../notes/lec_notes/lec8_notes#3. Bradley--Terry：把比较反馈变成可学习的 reward]] |
| §2.2(c) preference dataset 判断 | trajectory preference、标签噪声、held-out agreement | [[../notes/lec_notes/lec8_notes#3. Bradley--Terry：把比较反馈变成可学习的 reward]] |
| §2.2(f) reward 是否可恢复 | reward non-identifiability、preference 只约束相对差异 | [[../notes/lec_notes/lec7_notes#5. 从示范反推奖励：feature matching 与 MaxEnt IRL]]、[[../notes/lec_notes/lec8_notes#2. 从示范/比较到 preference learning]] |
| §3 DPO 推导 | KL-regularized objective、closed-form policy、log-partition cancellation | [[../notes/lec_notes/lec8_notes#5. DPO：把 reward optimization 改写成 policy optimization]] |
| §4 PAC/样本复杂度 | Hoeffding、union bound、$\epsilon$-optimal action、$\delta$ failure probability | [[../notes/lec_notes/lec9_notes]] 中的 UCB/浓缩不等式部分；这是 Ass3 额外的 bandit/PAC 复习 |
| §5 stated vs revealed preference | stated/revealed preference、reward choice、探索与伦理 | [[../notes/lec_notes/lec10_notes]] 的 value alignment/ethics 部分；结合题面案例 |

### Ass3：代码部分

| 实现顺序 | 需要掌握的知识 | Lecture notes 位置 | 代码位置 |
|---|---|---|---|
| 1. Hopper PPO 基线 | PPO、reward terms、early termination、seed evaluation | [[../notes/lec_notes/lec6_notes#5. Proximal Policy Optimization]] | `ass3/assignment3_starter_code/ppo_hopper.py` |
| 2. 视频和曲线 | rollout、checkpoint、episodic return、标准误 | [[../notes/lec_notes/lec6_notes#6. Assignment Readiness：Assignment 2]] | `render.py`、`plot.py` |
| 3. Reward model | Bradley–Terry、trajectory reward sum、BCE loss、label conversion | [[../notes/lec_notes/lec8_notes#3. Bradley--Terry：把比较反馈变成可学习的 reward]] | `run_rlhf.py` 的 `RewardModel` |
| 4. RLHF | learned reward + PPO、KL 稳定性、reward hacking | [[../notes/lec_notes/lec8_notes#4. RLHF：用人类反馈训练语言模型]] | `run_rlhf.py` |
| 5. DPO data/policy | action sequence distribution、winner/loser、reference policy | [[../notes/lec_notes/lec8_notes#5.2 Bandit 形式的 DPO loss]]、[[../notes/lec_notes/lec8_notes#5.5 Assignment 3：从文本 DPO 到 Hopper action sequence]] | `run_dpo.py` |
| 6. SFT → DPO | supervised pretraining、冻结 reference、DPO update | [[../notes/lec_notes/lec8_notes#5. Assignment 3：从文本 DPO 到 Hopper action sequence]] | `run_dpo.py` 的 `SFT.update`、`DPO.update` |

---

## Assignment 2

题目：[[ass2/CS234_A2_Questions.pdf]]  
代码：`ass2/assignment2_starter_code/`  
模板：`ass2/assignment2_template/main.tex`

### A. DQN 书面题

对应题目：Ass2 §1.1(a)–(c)

需要掌握：

- tabular Q-learning 与 DQN 的区别：表格 Q 值、神经网络近似、minibatch 更新。
- replay buffer 的作用：打破连续样本相关性、重用经验、改善数据效率。
- target network：固定一段时间的 bootstrap target，减轻 target 漂移。
- epsilon-greedy 探索和 terminal transition 的 target 处理。
- function approximation、状态空间大或连续时为什么表格方法会失效。

复习：[[../notes/lec_notes/lec4_notes]]，重点看 DQN、experience replay、fixed Q-targets 和 tabular control。

### B. REINFORCE 与 policy gradient

对应题目：Ass2 §2.1、§2.7(a)，代码中的 policy-gradient 基础。

需要掌握：

- policy gradient theorem：

  $$
  \nabla_\theta J(\theta)=\mathbb E[\nabla_\theta\log\pi_\theta(a\mid s)Q^{\pi_\theta}(s,a)]
  $$

- likelihood-ratio / score-function identity。
- trajectory、reward-to-go 和 Monte Carlo return：

  $$
  G_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}
  $$

- 用从后往前的递推在 $O(T)$ 时间计算全部 returns。
- categorical policy 与 Gaussian policy 的区别。
- `log_prob` 的 batch shape：连续动作的 action 维需要聚合成每个样本一个标量。

复习：[[../notes/lec_notes/lec5_notes]]，重点看 policy gradient、softmax/Gaussian policy、REINFORCE 和 reward-to-go。

代码映射：

- `code/network_utils.py`：用 PyTorch 构造 MLP。
- `code/policy.py`：动作分布、采样、log-probability。
- `code/policy_gradient.py`：policy 初始化、returns、advantage normalization、policy update。

### C. Baseline 与 advantage

对应题目：Ass2 §2.2–§2.3，代码中的 baseline network。

需要掌握：

- action-independent baseline 不改变 policy gradient 的期望。
- baseline 的实际作用是降低 Monte Carlo return 的方差。
- 状态价值 baseline：$b_\phi(s)\approx V^\pi(s)$。
- advantage：

  $$
  \hat A_t=G_t-b_\phi(s_t)
  $$

- baseline 的 MSE 回归目标。
- advantage normalization 的均值、标准差和数值稳定性。

复习：[[../notes/lec_notes/lec5_notes#Assignment Readiness：Assignment 2]]、[[../notes/lec_notes/lec6_notes#2.3 Vanilla policy gradient：把 policy 与 baseline 接成训练循环]]。

代码映射：`code/baseline_network.py`。

### D. PPO

对应题目：Ass2 §2.4、§2.5、§2.7(b)–(c)。

需要掌握：

- on-policy 与有限数据复用的区别。
- old policy、current policy、importance ratio：

  $$
  z_\theta(s,a)=\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{old}}(a\mid s)}
  $$

- 用 log-probability 差计算 ratio：`exp(new_log_prob - old_log_prob)`。
- clipped surrogate objective，以及正/负 advantage 时的分段行为。
- 为什么 clipping 会产生零梯度区域。
- 为什么 REINFORCE 不需要缓存 old log-probability，而 PPO 需要。
- PPO 中 advantage 和 old log-probability 在一次 batch 更新期间应保持固定。

复习：[[../notes/lec_notes/lec6_notes#5. Proximal Policy Optimization]]、[[../notes/lec_notes/lec7_notes#Assignment Readiness]]。

代码映射：`code/ppo.py` 的 `PPO.update_policy`。

### E. Ass2 分布与性能差异证明

对应题目：Ass2 第 3 题。

需要掌握：

- trajectory distribution $\rho^\pi(\tau)$ 的分解：初始状态、policy action probability、transition probability。
- timestep state distribution $p^\pi(s_t=s)$。
- discounted stationary state distribution $d^\pi(s)$。
- discounted visitation identity。
- advantage $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$。
- performance difference lemma，以及 tower property 和 telescoping cancellation。

复习：[[../notes/lec_notes/lec6_notes#4. 从性能差异恒等式到可优化 surrogate]]。

### F. Ass2 实验前检查

- [ ] `build_mlp` 输出 shape 正确。
- [ ] CartPole categorical policy 可以采样 action 和 log-probability。
- [ ] Pendulum/Cheetah Gaussian policy 的 log-probability 已沿 action 维求和。
- [ ] `get_returns` 使用反向递推，而非 $O(T^2)$ 双重循环。
- [ ] baseline、advantage、PPO ratio 的 shape 都是 flattened batch 的 `[batch]`。
- [ ] 先跑 CartPole，再跑 Pendulum，最后跑 Cheetah。
- [ ] 每个环境比较 no-baseline、baseline、PPO，并记录 seed、平均回报、训练时间和曲线。

---

## Assignment 3

题目：[[ass3/CS234_A3_Questions.pdf]]  
代码：`ass3/assignment3_starter_code/`  
模板：`ass3/assignment3_template/hw3_template.tex`

### A. Hopper reward engineering

对应题目：Ass3 §1.1(a)–(c)、§1.2(d)–(f)。

需要掌握：

- reward engineering：任务目标、奖励项、权重和终止条件之间的关系。
- reward misspecification、reward hacking 和 specification gaming。
- Hopper 的 forward progress、control cost、healthy reward/termination 等奖励组成。
- early termination 的优点：避免无意义或危险状态继续采样；缺点：可能截断探索、改变优化目标。
- PPO 训练曲线、seed variance、standard error 和 wall-clock time。
- 通过 rollout 视频检查策略是否真的学会“跳跃前进”，而不只是获得高分的异常行为。

复习：[[../notes/lec_notes/lec7_notes#7. Assignment Readiness]] 的 reward/IRL 边界，再结合题面和 MuJoCo Hopper 文档阅读具体奖励项。

代码入口：`ppo_hopper.py`、`plot.py`、`render.py`。

### B. Behavior cloning、IRL 与 preference learning 背景

需要掌握：

- behavior cloning：用专家 state-action 对监督学习 policy。
- compounding errors 和 train/test state-distribution mismatch。
- DAGGER 如何在 learner-induced states 上查询专家并聚合数据。
- inverse RL 与 behavior cloning 的区别：前者学习 reward，后者直接拟合 action。
- reward 的非唯一性：偏好数据通常只能确定相对比较，不能保证恢复真实 reward。

复习：[[../notes/lec_notes/lec7_notes#4. 模仿学习：从 action 标签到状态分布修复]]、[[../notes/lec_notes/lec7_notes#5. 从示范反推奖励：feature matching 与 MaxEnt IRL]]。

### C. Bradley–Terry preference model 与 reward model

对应题目：Ass3 §2.1、§2.2(c)–(f)。

需要掌握：

- 两条 trajectory 的累计预测 reward。
- Bradley–Terry preference probability：

  $$
  P(\sigma^1\succ\sigma^2)=\sigma(R_\theta(\sigma^1)-R_\theta(\sigma^2))
  $$

- binary cross-entropy preference loss。
- 对 reward difference 求梯度，并区分 winner/loser 的符号。
- preference label 的编码转换：Assignment 3 数据中 `0` 表示第一条胜出，`1` 表示第二条胜出，`0.5` 表示平局；不能直接套用讲义中常见的 `mu=1` 表示第一条胜出约定。
- reward model 预测准确不等于恢复真实人类价值；需要 held-out preference 检查。

复习：[[../notes/lec_notes/lec8_notes#3. Bradley--Terry：把比较反馈变成可学习的 reward]]。

代码入口：`run_rlhf.py` 中的 `RewardModel`。

### D. RLHF

对应题目：Ass3 §2.2(d)–(g)。

需要掌握：

- preference dataset → reward model → PPO policy 的两阶段 pipeline。
- learned reward 与 ground-truth reward 的区别。
- reward model error 如何被 PPO 放大为 reward hacking。
- KL-regularized policy optimization：在偏好 reward 与 reference policy 稳定性之间权衡。
- original reward、learned reward、held-out preference accuracy 和 rollout behavior 的比较。

复习：[[../notes/lec_notes/lec8_notes#4. RLHF：用人类反馈训练语言模型]]。

代码入口：`run_rlhf.py`、`data/long-prefs-hopper.npz`、`data/prefs-hopper.npz`。

### E. DPO

对应题目：Ass3 §3。

需要掌握：

- DPO 不显式训练 reward model，但仍需要 preference pairs 和冻结 reference policy。
- winner/loser action sequence 的联合 log-probability。
- reference log-probability 必须在 `torch.no_grad()` 下计算。
- DPO loss：

  $$
  \mathcal L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right]\right)
  $$

- KL-regularized objective 的 closed-form policy 与 log-partition cancellation。
- SFT、DPO 和 RLHF 的数据需求与优化对象差异。
- action sequence 不是完整 open-loop episode；作业使用 receding-horizon control，每次只执行计划的一步再重新规划。

复习：[[../notes/lec_notes/lec8_notes#5. DPO：把 reward optimization 改写成 policy optimization]]。

代码入口：`run_dpo.py` 中的 `ActionSequenceModel`、`SFT.update`、`DPO.update`。

### F. Ass3 实验前检查

- [ ] 先跑 Hopper PPO，确认 early termination 开关能工作。
- [ ] 检查 preference dataset 的 shape、label 编码和 trajectory 长度。
- [ ] 先测试 reward model 的 trajectory reward 和 preference probability，再训练 PPO。
- [ ] 检查 reward model 的 held-out preference 表现，再解释 learned-reward PPO 曲线。
- [ ] 检查 DPO 每个 action sequence 最终得到一个标量 log-probability。
- [ ] 确认 reference policy 参数不更新。
- [ ] 对 RLHF、DPO 结果同时看曲线和视频，不只比较最终 reward。

---

## 复习时的最低掌握标准

开始写某个函数前，至少能够：

1. 说清楚输入、输出和每个 tensor 的 shape。
2. 写出该函数对应的数学公式。
3. 用一个长度为 3 或 4 的 toy batch 手算一次结果。
4. 解释该函数为什么使用当前 policy、old policy、baseline 或 reference policy。
5. 跑一个最小 smoke test，再开始完整训练。

> [!warning] 作业诚信
> 课程要求提交内容由本人完成。复习笔记、公式推导和调试可以使用课程资料与讨论，但提交代码和文字应由自己理解并独立整理。
