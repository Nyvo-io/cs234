---
title: CS234 Lecture 5 - Policy Gradient I
aliases:
  - CS234 Lec5
tags:
  - cs234
  - reinforcement-learning
  - policy-gradient
---

# CS234 Lecture 5 Notes: Policy Gradient I

来源：`lecture/lecture5post.pdf`，CS234 Winter 2026，Emma Brunskill，共 61 个物理 PDF 页面。本文以课堂 post deck 为覆盖依据；`lecture/lecture5pre.pdf` 第 62--78 页的预习内容不计入本讲覆盖，其中 vanilla policy gradient、actor-critic 与 target alternatives 在 Lecture 6 按课堂 post deck 正式讲解。

笔记规范：cs234-rl-tutor v2。下方覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：不适用（延伸阅读只列课件指定教材与经典来源，不包含时效性结论）。

## 0. 本讲覆盖清单

- [x] 第 1--4 页：TD / bootstrapping 复习、课程结构与本讲位置；写入 §2.1。
- [x] 第 5--21 页：函数逼近控制、deadly triad、DQN replay / fixed targets、Atari 与消融结果；作为 Lecture 4 回引写入 §2.1。
- [x] 第 22--31 页：policy-based RL 动机、方法分类、随机策略、Aliased Gridworld 与直接策略优化；写入 §2.2--§2.4。
- [x] 第 32--40 页：参数化策略价值、轨迹概率、likelihood-ratio 推导与 dynamics 消项；写入 §3。
- [x] 第 41--50 页：score function、softmax / Gaussian policy、直觉测验与 policy gradient theorem；写入 §3.2 与 §4。
- [x] 第 51--58 页：时间因果结构、reward-to-go 与 REINFORCE；写入 §5.1--§5.2。
- [x] 第 59--60 页：梯度估计器的评价目标与章节导航；写入 §5.3，导航页不单列教学内容。
- [x] 第 61 页：state baseline 的作用、无偏结论与近似选择；写入 §5.4，并把完整方差分析留给 Lecture 6。
- [x] Assignment 2 的 REINFORCE、baseline、policy distribution 与真实 starter-code shape 已映射到 §6。

**视觉材料决策**：第 29 页 Aliased Gridworld 的最终动画帧直接嵌入 §2.3；第 8、18、19 页的 DQN 流程和 Atari 结果属于 Lecture 4 已完整讲解的重复视觉，因此回引而不重复嵌入；第 20 页消融表用 §2.1 的文字结论覆盖。其余公式动画页用最终公式和可搜索推导重写，避免嵌入难以检索的整页截图。

## 1. 本讲主线

Lecture 4 学的是“先估计 $Q$，再从 $Q$ 间接产生策略”。Lecture 5 改变了优化对象：直接用参数 $\theta$ 表示动作分布 $\pi_\theta(a\mid s)$，让期望回报对 $\theta$ 做梯度上升。

整条推理链是：

1. 把策略目标写成所有轨迹的加权和 $J(\theta)=\sum_\tau P(\tau;\theta)R(\tau)$。
2. 用 likelihood-ratio identity 把难求的 $\nabla_\theta P(\tau;\theta)$ 改写成 $P(\tau;\theta)\nabla_\theta\log P(\tau;\theta)$。
3. 分解轨迹概率后，初始状态分布和环境 dynamics 都与 $\theta$ 无关，只剩每一步的 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。
4. 用因果结构删掉动作之前的奖励，得到 reward-to-go 版本的 REINFORCE。
5. 用 action-independent baseline 把动作结果改写成“相对当前状态通常表现的偏差”，在不改变期望梯度的前提下降低方差；更完整的方差分析与 actor-critic 留到 Lecture 6。

最核心的一句话是：**提高一条高回报轨迹中已选动作的概率，降低一条低于预期轨迹中已选动作的概率；更新强度由 return 或 advantage 决定。**

## 2. 从价值函数到直接策略优化

### 2.1 Lecture 4 过渡：DQN 与 deadly triad

*首次完整讲解：Lecture 4 §5.6「Deadly triad」、§6.1「DQN」、§6.2「Experience replay」和 §6.3「Target network」。本节只补充：这些 value-based 方法与直接策略搜索的分界。*

第 2--3 页的复习题把两个旧概念重新对齐：TD update 让当前估计朝 $r_t+\gamma V(s_{t+1})$ 移动；bootstrapping 指 target 使用下一状态的价值估计，而不是已经知道的真实价值。它们已在 Lecture 3 §3.1 和 §4.5 完整讲解。

课件第 5--21 页随后收尾 model-free value-based RL：

- MC 函数逼近用完整回报 $G_t$ 作为 $Q^\pi(s_t,a_t)$ 的样本 target；
- SARSA 用实际下一动作的 target $r_t+\gamma\hat Q(s_{t+1},a_{t+1};w)$；
- Q-learning 用 $r_t+\gamma\max_{a'}\hat Q(s_{t+1},a';w)$；
- function approximation、bootstrapping、off-policy learning 同时出现时会形成 deadly triad；
- DQN 用 experience replay 减弱连续样本相关性并重复利用旧 transition，用固定参数 $w^-$ 计算一段时间内稳定的 target。

DQN 仍然先学习 $\hat Q(s,a;w)$，动作策略通常由 epsilon-greedy 间接产生。本讲开始不再要求先有 $Q$ 才能定义策略，而是直接学习 $\pi_\theta(a\mid s)$。

Fixed target network 需要额外保存一套参数，因此增加的主要是网络参数内存；target network 只负责生成 target，并不意味着环境交互和所有计算都严格翻倍。课件还用 Atari 消融表说明 replay 与 fixed targets 的经验作用，并要求能实现 TD(0)、MC、Q-learning、MC control，能定性解释 deadly triad 和 DQN 的两个关键稳定化组件。这些都是 Lecture 4 的复习，本讲不重复完整推导。



### 2.2 Value-based 与 policy-based；actor-critic 只作路线图

*首次完整讲解：Lecture 5 §2.2「Value-based 与 policy-based；actor-critic 只作路线图」。*

**基于策略的强化学习（policy-based RL)**  直接用参数表示策略：

$$
\pi_\theta(a\mid s)=P(a\mid s;\theta).
$$

这个对象的输入是状态 $s$，输出是动作上的概率分布，在状态 s 下，策略以多大概率选择动作 a；$\theta$ 是需要优化的参数。
三类方法的区别在于显式学习什么：

| 方法 | 显式学习的对象 | 如何产生动作 |
|---|---|---|
| Value-based | $V$ 或 $Q$ | 从价值函数间接产生，如 epsilon-greedy |
| Policy-based | $\pi_\theta$ | 直接从策略分布采样 |
| Actor-critic | $\pi_\theta$ 和 $V_w$ 或 $Q_w$ | actor 决策，critic 评价 |

> [!important] 两条独立的判断轴
> “策略是否显式参数化”与“是否显式学习价值函数”不是同一件事。纯 policy-based 方法显式学习 $\pi_\theta$、可以不学习 value；value-based 方法显式学习 value、策略通常是隐式的；actor-critic 同时显式学习二者。后面比较算法时，先分别回答这两个问题。

这里先把 actor-critic 当作分类图中的路线提示；它的两个组件和更新关系在 Lecture 6 §2.4 首次完整讲解。

### 2.3 为什么需要随机策略：Aliased Gridworld

*首次完整讲解：Lecture 5 §2.3「为什么需要随机策略：Aliased Gridworld」。*

**状态混叠（state aliasing)**  是一种表示问题：多个真实环境状态被映射成同一个 observation 或 feature representation，导致智能体无法根据当前输入区分它们。课件中的两个灰色格子就是如此。若特征只记录“北面有墙且动作是向东”等局部信息，那么两个真实位置会得到相同表示；这不是环境真的只有一个状态，而是当前表示丢失了区分它们所需的信息。

确定性策略只能在两个灰色格子都向东，或都向西，因此可能反复被困在走廊里。随机策略却能在这类观测下令：

$$
\pi_\theta(\text{East}\mid \text{aliased state})=0.5,
\qquad
\pi_\theta(\text{West}\mid \text{aliased state})=0.5.
$$

它会以较高概率最终走出走廊。这里的重点不是“随机总比确定好”，而是：**当状态表示发生 aliasing，最优的可表示策略可能本来就是随机的。** 直接策略参数化能自然表达这种解。

![[lec5-aliased-gridworld-p29.png|900]]

*图：aliased gridworld 中两个灰色位置共享同一表示；随机策略在两处都以 $0.5$ 向东或向西，因而能以高概率离开走廊。来源：`lecture/lecture5post.pdf`，物理 PDF 第 29 页。*

课件还把早期 AIBO 步态优化、端到端 visuomotor policy、NLP sequence-level training、ChatGPT 训练和 Assignment 2 的 PPO 作为 policy gradient 的应用动机；这些只说明方法影响范围，PPO 的目标函数不属于本讲推导范围。


### 2.4 直接策略优化与 policy gradient

*首次完整讲解：Lecture 5 §2.4「直接策略优化与 policy gradient」。*

正式引入 **策略梯度（Policy Gradient）**：不再先学 V 或 Q，再从价值函数里挑动作，而是**直接调整策略本身的参数，让策略获得更高回报**

直接策略搜索本质上是一个优化问题。课件指出它既可以使用 [gradient-free optimization](academic-term-lookup:gradient-free%20optimization)，也可以使用 gradient、conjugate gradient 或 quasi-Newton 方法；
本讲选择梯度路线，是因为可利用序列决策中的概率结构得到更高效的更新。

固定 episodic MDP 与初始状态分布 $\mu$(每个 episode 开始时，初始状态 s0​ 可能不完全一样)。课件使用固定 $s_0$；这里写成更一般但等价的标量目标：

$$
J(\theta)
=\mathbb E_{s_0\sim\mu}\!\left[V^{\pi_\theta}(s_0)\right].
$$


若 $\mu$ 把全部概率放在一个状态上，初始状态始终一致，就退化为课件的 $V^{\pi_\theta}(s_0)$。


$V^{\pi_\theta}(s_0)$：执行当前策略后的==期望总回报==。当前参数为 θ 的策略，从初始状态出发，平均能获得多少累计奖励

**策略梯度（policy gradient）**是标量目标 $J(\theta)$ 对参数向量 $\theta\in\mathbb R^d$ 的梯度：

$$
\nabla_\theta J(\theta)
=
\begin{bmatrix}
\partial J/\partial\theta_1\\
\vdots\\
\partial J/\partial\theta_d
\end{bmatrix}.
$$

因为目标是最大化价值，参数沿梯度方向做梯度上升（gradient ascent）：

$$
\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta),
$$

其中 $\alpha>0$ 是学习率。深度学习库的 optimizer 通常最小化 loss，所以实现时常取 $L(\theta)=-J(\theta)$。我们这里要最大化J

> [!example] 贯穿计算：一步两动作策略
> Episode 只有一步，动作 $L$、$R$ 的奖励分别为 $1$ 和 $4$，策略满足 $\pi_\theta(R)=\sigma(\theta)$、$\pi_\theta(L)=1-\sigma(\theta)$。于是
>
> $$
> J(\theta)
> =\bigl(1-\sigma(\theta)\bigr)\times1
> +\sigma(\theta)\times4
> =1+3\sigma(\theta).
> $$
>
> 当 $\theta_0=0$ 时，$J(0)=2.5$，真实梯度为
>
> $$
> \nabla_\theta J(0)
> =3\sigma(0)\bigl(1-\sigma(0)\bigr)
> =0.75.
> $$
>
> 若 $\alpha=0.1$，使用真实梯度更新一次得到 $\theta_1=0.075$，动作 $R$ 的概率从 $0.5$ 提高到 $\sigma(0.075)\approx0.519$。后续各节沿用同一个例子，逐步回答在不知道环境期望时如何从 sampled trajectories 估计这个 $0.75$。

梯度只提供当前参数附近的上升方向；神经网络目标通常非凸，因此 policy gradient 通常只期望到达局部最优或驻点，不保证全局最优。

## 3. 从轨迹目标到 dynamics-free 梯度

### 3.1 轨迹分布与策略目标

*首次完整讲解：Lecture 5 §3.1「轨迹分布与策略目标」。*

本文约定在 $s_t$ 执行动作 $a_t$ 后观察奖励 $r_t$ 和下一状态 $s_{t+1}$。一条有 $T$ 次决策的轨迹（trajectory）写成：

$$
\tau=(s_0,a_0,r_0,\ldots,s_{T-1},a_{T-1},r_{T-1},s_T).
$$

本节先沿用课件的未折扣记号：

$$
R(\tau)=\sum_{t=0}^{T-1}r_t.
$$

在课件固定起始状态 $s_0$ 的特例下，同一个目标也可从首个动作分解为

$$
J(\theta)=\sum_a\pi_\theta(a\mid s_0)Q^{\pi_\theta}(s_0,a),
$$
==策略 $\pi_\theta$并不是提前把整条轨迹中的动作全部定死，而是对每个状态 s，给出各个动作的选择概率 $\pi_\theta(a\mid s)$，那个符号形式上是求和，本质上是期望。把每个动作的 Q 值，按照策略选择它的概率进行加权平均==


但这里的 $Q^{\pi_\theta}$ 本身也随策略改变。课件接下来选择轨迹表示，把所有随机来源和可求导的 policy 因子显式展开。

**轨迹分布（trajectory distribution）** $P(\tau;\theta)$ 是执行策略 $\pi_\theta$ 后这条轨迹出现的概率。
它不是单步策略概率；它同时包含初始状态、每一步动作概率和环境转移概率：

$$
P(\tau;\theta)
=
\mu(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

因此策略目标可写为期望形式：

$$
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
=
\sum_\tau P(\tau;\theta)R(\tau).
$$

这里的
$$\mathbb E_{\tau\sim \pi_\theta}$$

表示：按照当前策略 π与环境交互时，可能走出的轨迹不止一条，因此要对不同轨迹产生的结果取期望。

注意不是简单地把所有轨迹等权平均，而是：

> **每条轨迹按照它在当前策略下出现的概率 $P_\theta(\tau)$加权。**

R(τ) 表示这条完整轨迹获得的累计回报。


期望为什么可以写成求和：
期望本质上就是：$\text{结果的概率}\times\text{结果的数值}$，然后把所有可能结果加起来。


沿用 §2.4 的两动作例子。两条可能轨迹分别记作 $\tau_L$、$\tau_R$，其概率为

$$
P(\tau_R;\theta)=\sigma(\theta),
\qquad
P(\tau_L;\theta)=1-\sigma(\theta).
$$

在 $\theta=0$ 时，轨迹和写法给出

$$
J(0)=P(\tau_L;0)\times1+P(\tau_R;0)\times4
=0.5\times1+0.5\times4=2.5.
$$

这里的新信息不是再次求目标值，而是确认 $P(\tau;\theta)$ 如何进入轨迹加权和。策略优化改变的不是动作对应的奖励，而是两条轨迹未来被采样到的概率。后续 likelihood-ratio、REINFORCE 和 baseline 都继续使用这个例子。

> [!important] 单步概率与整条轨迹概率
> $\pi_\theta(a_t\mid s_t)$ 只回答“在当前状态选这个动作的概率”；$P(\tau;\theta)$ 回答“整条状态--动作序列发生的概率”。前者是后者乘积中的一个因子，不能互换。正因为对数把乘积变成求和，整条轨迹的 score 才能拆成逐时间步 policy score。

### 3.2 似然比恒等式与 score function

*首次完整讲解：Lecture 5 §3.2「似然比恒等式与 score function」。*

直接对目标求导：

$$
\nabla_\theta J(\theta)
=
\sum_\tau \nabla_\theta P(\tau;\theta)R(\tau).
$$

困难在于 $\nabla_\theta P(\tau;\theta)$ 仍然作用在完整轨迹概率上，里面包含很多步策略概率和环境转移概率。**似然比恒等式（likelihood-ratio identity）**把它改写为“轨迹概率乘以 log-probability 的梯度”：

$$
\nabla_\theta P(\tau;\theta)
=
P(\tau;\theta)
\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}
=
P(\tau;\theta)\nabla_\theta\log P(\tau;\theta),
$$

代回目标后得到：
引入 log 后，梯度可以写成一个期望

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
R(\tau)\nabla_\theta\log P(\tau;\theta)
\right].
$$



**得分函数（score function）**是参数化概率的对数对参数的梯度：

$$
\nabla_\theta\log p(x;\theta).
$$

它是与 $\theta$ 同维度的向量，描述“沿哪个参数方向移动会增加样本 $x$ 的 log-probability”。它不是 reward，也不是概率本身。

这个恒等式并不只属于 MDP，也不要求被加权的函数可导：只要能够从可微的参数化分布 $p(x;\theta)$ 采样、在其支持集上计算 score，并且 $f(x)$ 不直接依赖 $\theta$，就能用 $f(x)\nabla_\theta\log p(x;\theta)$ 估计相应期望的梯度；$f$ 可以未知、离散或不连续。RL 中只是令 $x=\tau$、$f(x)=R(\tau)$。


从当前策略采样 $m$ 条独立轨迹后，Monte Carlo 梯度估计量是(写成了期望的形式，为什么可以这么写，下面有讲解)：

$$
\hat g
=
\frac{1}{m}
\sum_{i=1}^m
R(\tau^{(i)})
\nabla_\theta\log P(\tau^{(i)};\theta).
$$

> [!example] 具体计算：用两条轨迹恢复真实梯度
> 沿用 §2.4 的两动作策略。在 $\theta=0$ 时，两个动作的 score 分别为
>
> $$
> \nabla_\theta\log\pi_\theta(R)
> =1-\sigma(0)=0.5,
> \qquad
> \nabla_\theta\log\pi_\theta(L)
> =-\sigma(0)=-0.5.
> $$
>
> $R$ 和 $L$ 各以 $0.5$ 的概率出现，因此 likelihood-ratio 形式给出
>
> $$
> \begin{aligned}
> \nabla_\theta J(0)
> &=0.5\times4\times0.5
>   +0.5\times1\times(-0.5)\\
> &=0.75.
> \end{aligned}
> $$
>
> 结果与直接对 $J(\theta)=1+3\sigma(\theta)$ 求导完全一致。这里相等的是所有可能轨迹的期望；单条 sampled trajectory 仍然只是 noisy estimate。

Reward 不需要可导，因为求导发生在概率模型上，而不是环境反馈上。



---

具体讲解利用等式取log作用：

取log后变化为：
$$\nabla_\theta J(\theta) = \sum_\tau R(\tau) P(\tau;\theta) \nabla_\theta\log P(\tau;\theta)$$

调整顺序：$\nabla_\theta J(\theta) = \sum_\tau P(\tau;\theta) \left[ R(\tau)\nabla_\theta\log P(\tau;\theta) \right]$

而“概率乘以某个量再求和”就是期望：

$$\boxed{ \nabla_\theta J(\theta) = \mathbb E_{\tau\sim\pi_\theta} \left[ R(\tau)\nabla_\theta\log P(\tau;\theta) \right] }$$​

这就是第一个关键作用：

> 引入 log 后，梯度可以写成一个期望，于是可以通过采样 trajectory 来近似，而不用枚举所有可能轨迹。


例如采样 m 条轨迹：$\hat g = \frac1m \sum_{i=1}^m R(\tau^{(i)}) \nabla_\theta\log P(\tau^{(i)};\theta)$

[!question]
	为什么可以写成期望：

 **从策略中采样出来的轨迹，本身就是按照 $P(\tau;\theta)$的概率出现的。**

因此，想用“采样平均”估计一个量，它必须先写成这种形式：

$$\sum_\tau P(\tau;\theta)\,f(\tau) = \mathbb E_{\tau\sim P}[f(\tau)]$$

而原来的梯度刚开始并不是这种形式。

所以我们不需要同时列出所有轨迹，只需要不断：

1. 用当前策略跑一个 episode，得到 $\tau^{(i)}$；
2. 计算它的回报 $R(\tau^{(i)})$；
3. 计算它的 $\nabla_\theta\log P(\tau^{(i)};\theta)$；
4. 对多条轨迹的结果求平均。

---


更重要的一点是：
	log 可以把连乘变成求和


一条轨迹：$\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)$

它的概率是：

$P(\tau;\theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t) P(s_{t+1}\mid s_t,a_t)$

这是很多概率的连乘。

直接对这一大串乘积求导会很麻烦。取对数后：

$\log P(\tau;\theta) = \log\mu(s_0) + \sum_{t=0}^{T-1} \log\pi_\theta(a_t\mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1}\mid s_t,a_t)$

因为：$\log(xyz)=\log x+\log y+\log z$

于是“连乘”变成了“求和”。

再对 θ求梯度：

$$\nabla_\theta\log P(\tau;\theta) = \nabla_\theta\log\mu(s_0) + \sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t) + \sum_t\nabla_\theta\log P(s_{t+1}\mid s_t,a_t)$$

但初始状态分布和环境转移通常不依赖策略参数 θ，所以：

$\nabla_\theta\log\mu(s_0)=0$      $\nabla_\theta\log P(s_{t+1}\mid s_t,a_t)=0$

最终只剩：

$$\boxed{ \nabla_\theta\log P(\tau;\theta) = \sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t) }$$

代回去：

$$\boxed{ \nabla_\theta J(\theta) = \mathbb E_{\tau\sim\pi_\theta} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log\pi_\theta(a_t\mid s_t) \right] }$$​

这就是最基本的 ==REINFORCE 形式==。

所以出现了求和形式


### 3.3 为什么不需要对环境 dynamics 求导

将 §3.1 的轨迹概率取对数：

$$
\log P(\tau;\theta)
=
\log\mu(s_0)
+
\sum_{t=0}^{T-1}
\left[
\log\pi_\theta(a_t\mid s_t)
+
\log P(s_{t+1}\mid s_t,a_t)
\right].
$$

课件假设 MDP dynamics 和初始状态分布不随 policy 参数 $\theta$ 改变。对上式求导时，这两类项都为 0：

$$
\nabla_\theta\log P(\tau;\theta)
=
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

因此，完整轨迹的 score 只剩 policy score 的时间求和。代回梯度目标可得：

$$
\nabla_\theta J(\theta)
=
\mathbb E_\tau
\left[
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$




这里的 [dynamics-free](academic-term-lookup:dynamics-free) 只表示：不需要知道或求导 $P(s'\mid s,a)$  (model free 的rl不会显式得知P，而是根据多次实际采样取平均值)；

算法仍然必须与环境交互，采样状态、动作和奖励。若环境本身也由 $\theta$ 参数化，上面的消项条件便不成立。



## 4. 策略参数化与 policy gradient theorem


前面只有抽象的策略

之前我们写：$\pi_\theta(a\mid s)$

它表示“参数为 θ的策略，在状态 s 选择动作 a 的概率”。

但这只是一个抽象符号，还没有说明：

- 参数 θ怎么影响动作概率？
- 怎么计算 $\pi_\theta(a\mid s)$？
- 怎么计算 $\nabla_\theta\log\pi_\theta(a\mid s)$？

这一页就是要把它具体化。

### 4.1 Softmax 策略

*首次完整讲解：Lecture 5 §4.1「Softmax 策略」。*

对离散动作 比如a∈{左、右、前进、停止}，**Softmax 策略（softmax policy）**先用特征 $\phi(s,a)\in\mathbb R^d$ 和参数 $\theta\in\mathbb R^d$ 产生 logit。Logit 是每个动作的未归一化实数分数，可以为任意实数，本身不是概率：

$$
h_\theta(s,a)=\phi(s,a)^\top\theta,
$$
其中：

- s：当前状态；
- a：某个候选动作；
- $\phi(s,a)$：状态—动作的特征；
- θ：需要学习的参数；
- $h_\theta(s,a)$：动作 a的分数，也叫 logit 或 preference。

例如当前状态下有三个动作，算出的分数可能是：

$$h_\theta(s,a_1)=2$$ $$h_\theta(s,a_2)=1$$$$h_\theta(s,a_3)=-1$$

这些数只是相对偏好，并不是概率，因为：

- 可以是负数；
- 加起来不等于 1；
- 可能大于 1。

再用softmax归一化成概率：

$$
\pi_\theta(a\mid s)
=
\frac{\exp(\phi(s,a)^\top\theta)}
{\sum_{a'}\exp(\phi(s,a')^\top\theta)}.
$$

它的 score function 是：

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=
\phi(s,a)
-
\sum_{a'}\pi_\theta(a'\mid s)\phi(s,a')
=
\phi(s,a)-\mathbb E_{a'\sim\pi_\theta}[\phi(s,a')].
$$

第一项是“实际选中动作的特征”，第二项是“当前策略下的平均动作特征”。

> [!example] 具体计算：Softmax score 改变哪个 logit
> 两个动作的特征分别为 $\phi(s,L)=[1,0]^\top$、$\phi(s,R)=[0,1]^\top$，当前概率为 $\pi(L\mid s)=2/3$、$\pi(R\mid s)=1/3$。若实际选中 $L$，则
>
> $$
> \begin{aligned}
> \nabla_\theta\log\pi(L\mid s)
> &=[1,0]^\top
> -\left(\frac23[1,0]^\top+\frac13[0,1]^\top\right)\\
> &=\left[\frac13,-\frac13\right]^\top.
> \end{aligned}
> $$
>
> 用正权重沿这个方向更新，会提高 $L$ 的相对 logit，同时降低 $R$ 的相对 logit；若梯度权重为负，方向则相反。

### 4.2 Gaussian 策略


连续动作不能用有限个 categorical 概率列举。课件使用一维 **Gaussian 策略(Gaussian policy)**：

$$
a\sim\mathcal N(\mu_\theta(s),\sigma^2),
\qquad
\mu_\theta(s)=\phi(s)^\top\theta.
$$

在状态s下，策略产生一个高斯分布：

$$\boxed{ a\sim\pi_\theta(\cdot\mid s) = \mathcal N\left( \mu_\theta(s), \sigma_\theta(s)^2 \right) }$$​

通常神经网络输入状态 s，

输出：$\mu_\theta(s),\qquad \sigma_\theta(s)$

然后从这个分布中采样动作：$a\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2)$

例如网络输出：

$\mu_\theta(s)=2,\qquad \sigma_\theta(s)=0.5$

策略这一次可能采样：a=2.31

他表示是根据策略从高斯分布中采样出来的具体动作值，下次可能是a=2.18

---

Gaussian 策略的具体公式：

一维情况下：

$$\pi_\theta(a\mid s) = \frac{1}{\sqrt{2\pi}\sigma_\theta(s)} \exp \left[ -\frac{ (a-\mu_\theta(s))^2 }{ 2\sigma_\theta(s)^2 } \right]$$

动作越接近均值：

$a\approx\mu_\theta(s)$

其概率密度通常越高。

动作离均值越远：

$|a-\mu_\theta(s)|\uparrow$

概率密度越低。


批注：
连续动作，比如 a=转向角度,  a=关节力矩,  a =移动速度

例如转向角可以是：$a\in[-30^\circ,30^\circ]$

不可能给每个实数动作都分配一个 Softmax 概率，因此通常让策略输出一个高斯分布：

$$\boxed{ \pi_\theta(a\mid s) = \mathcal N\left(a;\mu_\theta(s),\sigma_\theta^2(s)\right) }$$

然后从这个分布中采样动作：$a\sim\pi_\theta(\cdot\mid s)$

其中：
- $\mu_\theta(s)$：策略目前最倾向的动作；
- $\sigma_\theta(s)$：探索程度；σ越大，采样出来的动作越分散，探索越强；σ 越小，动作越接近均值，策略越确定。
- θ：控制均值和方差的网络参数。




若 $\sigma$ 固定，则对均值参数的 score function 为：

把高斯策略取对数：

$$\log\pi_\theta(a\mid s) = -\frac{(a-\mu_\theta(s))^2} {2\sigma_\theta(s)^2} -\log\sigma_\theta(s) -\frac12\log(2\pi)$$
再求导：
$$
\nabla_\theta\log\pi_\theta(a\mid s)
=
\frac{a-\mu_\theta(s)}{\sigma^2}\phi(s).
$$

πθ​(a∣s)在这里严格来说表示的是动作 a 附近的**概率密度**，而不是这个精确动作的概率。不然一条连续曲线找一个小点的概率无限接近于0


> [!example] 具体计算：Gaussian score 如何移动均值
> 令 $\phi(s)=2$、$\theta=1$、$\sigma=2$，因此均值 $\mu_\theta(s)=2$。若采样动作 $a=4$，则
>
> $$
> \nabla_\theta\log\pi_\theta(4\mid s)
> =\frac{4-2}{2^2}\times2
> =1.
> $$
>
> 用正权重更新会提高均值，使 $a=4$ 更可能；负权重则把均值推离该动作。

上式只计算固定 $\sigma$ 时的均值参数梯度。若标准差也可学习，还需要对方差参数求梯度。Assignment 2 使用 diagonal Gaussian，并以 `log_std` 表示各动作维度的可学习标准差；联合 log-probability 必须对动作维度求和，最终保持 `[batch]` 形状。


### 4.3 策略梯度定理

**策略梯度定理(policy gradient theorem)**  说明：对可微策略和合适的 episodic、average-reward 或 discounted objective，可以把策略梯度写成：score 与动作价值的期望。
$$\boxed{ \text{score} = \nabla_\theta\log\pi_\theta(a_t\mid s_t) }$$​
它不是 reward，也不是 Q，而是"怎样改变参数 θ，能提高这次选中动作 at 的概率"的方向。


沿用本讲有限 episode、$\gamma=1$ 的约定，一种不隐藏时间求和的写法是：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
Q^{\pi_\theta}(s_t,a_t)
\right].
$$

课件把状态访问分布和时间求和压缩进 $\mathbb E_{\pi_\theta}$。

它表达的是：

> 在每个时间步，根据动作 at 相对于当前策略的梯度方向 以及这个动作的价值 $Q^\pi(s_t,a_t)$，来调整策略参数。


$Q^{\pi_\theta}(s_t,a_t)$ 是期望动作价值，不是一条轨迹的 realized return；具体 objective 不同，状态访问权重、折扣因子和整体常数也可能不同。

不是一条轨迹的 realized return的意思是：

假设从同一个 (s,a) 出发，由于环境随机，可能出现三种结果：

	Gt=10,   Gt=4,   Gt=−2

对应概率分别是：

	0.5,0.3,0.2

那么：

$Q^\pi(s,a) = 0.5\times10+0.3\times4+0.2\times(-2) = 5.8$

这里：

- 某一条实际轨迹可能得到 Gt=10；
- 另一条可能得到 Gt=−2；
- 但真正的 $Q^\pi(s,a)$ 是平均值 5.8。

所以：

$$\boxed{G_t=\text{一次采样结果}}$$

$$\boxed{ Q^\pi(s_t,a_t) = \mathbb E[G_t\mid s_t,a_t] }$$​


所以，一般 MDP 存在后续随机性，这导致$Q^{\pi_\theta}(s_t,a_t)$ 在实际中无法直接读取，实际算法还必须决定如何从 [sampled trajectories](academic-term-lookup:sampled%20trajectories) 估计它。因为Q是多个G的期望，同一个 (s,a) 得到的的 $G_t$可能不一样

轨迹 likelihood-ratio 推导是 episodic 情形的一条直接路线，策略梯度定理则给出更一般的表述，两者不是互相竞争的算法。

接下来的 REINFORCE 选择用 realized reward-to-go 作为 $Q^{\pi_\theta}$ 的 Monte Carlo estimate。（REINFORCE：用实际 reward-to-go 估计）
reward-to-go 就是从当前时刻 t 开始，往后实际拿到的所有奖励：
$$\boxed{ G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k }$$​​
Monte Carlo 的思想：
$$Q^\pi(s,a) = \mathbb E[G_t\mid s,a] \approx \text{实际采样得到的 }G_t $$​​
所以在REINFORCE 里直接用 Gt 去代替 Q​


[!quention]
>	前面的 likelihood-ratio 推导又是什么？

前面推导过：

$$J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$$

通过 log trick：

$$\nabla_\theta J = \mathbb E \left[ R(\tau) \sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t) \right]$$

这条路线是：

$$\text{完整 trajectory} \rightarrow P(\tau) \rightarrow \log P(\tau) \rightarrow \text{策略梯度}$$

它是在 episodic 情况下从“整条轨迹概率”直接推出来的。

而策略梯度定理写：
$$\nabla_\theta J = \mathbb E \left[ \sum_t Q^\pi(s_t,a_t) \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$$

它从 state-action value 的角度给出了一个更一般的统一结构。

所以课件才强调：这两者不是两个不同算法

它们只是**不同的数学表达/推导路线**。



---

真正的“算法选择”发生在哪里？

真正形成 REINFORCE，是当我们面对：$Q^\pi(s_t,a_t)$

问：

> 我实际怎么获得这个东西？

REINFORCE 选择：

$\boxed{ Q^\pi(s_t,a_t) \approx G_t }$

也就是通过完整 episode 的实际未来奖励来估计。


而别的算法可能做不同选择。

例如 Actor-Critic 会说：

> 我不想每次都等 episode 跑完，我训练一个 Critic 来估计价值。

例如：$Q_w(s,a)\approx Q^\pi(s,a)$

或者通过：$\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$估计 advantage。


总结：

$\boxed{ \text{梯度} = \text{score}\times Q^\pi }$

策略梯度定理告诉你“应该拿 Qπ 给动作加权”；REINFORCE 告诉你“真实 Qπ不知道，就跑一条完整轨迹，用实际的后续累计奖励 Gt当作它的 Monte Carlo 样本”。

[!important]score 有一个非常重要的性质:

>	在固定状态 s 下：$\boxed{ \mathbb E_{a\sim\pi_\theta} [ \nabla_\theta\log\pi_\theta(a|s) ] =0 }$


对于离散动作：

$E[\nabla\log\pi] = \sum_a \pi(a|s) \nabla_\theta\log\pi(a|s)$

而：

$\nabla\log\pi = \frac{\nabla\pi}{\pi}$

所以：

$= \sum_a \pi(a|s) \frac{\nabla\pi(a|s)}{\pi(a|s)}$

约掉：

$= \sum_a\nabla\pi(a|s)$ 
$= \nabla\sum_a\pi(a|s)$

因为：$\sum_a\pi(a|s)=1$

所以：$\boxed{ \mathbb E[\text{score}]=\nabla 1=0 }$



## 5. [REINFORCE](academic-term-lookup:reinforce) 与方差降低

### 5.1 利用因果结构：从整条return到 reward-to-go

课件第 52--56 页先取 $\gamma=1$。若仍把整条轨迹回报乘到每一个动作的 score 上，时刻 $t$ 的动作会被 $r_0,\ldots,r_{t-1}$ 加权，但这些奖励在动作 $a_t$ 发生前已经确定，不可能由 $a_t$ 导致。

令 $H_t$ 表示选择 $a_t$ 前已经发生的历史。Ht​=(s0​,a0​,r0​,…,st​)。 
对任意 $t'<t$，过去奖励 $r_{t'}$ 在给定 $H_t$ 后是已知量，而当前动作 score 的条件期望为 0：

$$
\begin{aligned}
&\mathbb E
\left[
r_{t'}\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]\\
&=\mathbb E
\left[
r_{t'}
\mathbb E_{a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\mid H_t
\right]
\right]
=0.
\end{aligned}
$$

这就是前面刚证明的：

$$\mathbb E_{a_t\sim\pi} [ \nabla\log\pi(a_t|s_t) ] =0$$

因此：
$$= \mathbb E[r_{t'}\times0]=0$$


因此，所以过去奖励对**期望梯度**没有贡献，只会增加样本方差(因为某一次实际采样中，它一般不等于 0。只有大量样本平均后才抵消为0)。

每个动作只需保留它之后的 **reward-to-go（从当前时刻起的回报）**：

$$
G_t=\sum_{t'=t}^{T-1}r_{t'}.
$$

得到：(把整条 return 转变成了 ==reward-to-go==)

$$
\nabla_\theta J(\theta)
=
\mathbb E_\tau
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right].
$$

因为：

假设 γ=1：

$R(\tau) = r_0+r_1+\cdots+r_{T-1}$

对于 t 时刻，可以拆成：

$R(\tau) = \underbrace{r_0+\cdots+r_{t-1}}_{\text{动作 }a_t\text{ 之前}} + \underbrace{r_t+r_{t+1}+\cdots}_{G_t}R(τ)$

也就是：

$R(\tau) = R_{\text{past}}+G_t$
所以：

$\mathbb E [ \nabla\log\pi(a_t|s_t)R(\tau) ]$

变成：

$\mathbb E [ \nabla\log\pi(a_t|s_t)R_{\text{past}} ] + \mathbb E [ \nabla\log\pi(a_t|s_t)G_t ]$


为什么前面的奖励可以直接删掉？

因为 at **不可能影响过去已经发生的奖励**。数学证明上面也证明了,socre=0

==这就得到了上面的REINFORCE 形式。==



Assignment 2 使用折扣版本：

有折扣因子时：

$$
G_t
=
\sum_{k=0}^{T-1-t}\gamma^k r_{t+k}
=
r_t+\gamma G_{t+1}.
$$

反向递推从 $G_T=0$ 开始即可在 $O(T)$ 时间算出全部 $G_t$；分别从每个 $t$ 向后重算会是 $O(T^2)$。
从后往前算时间复杂度更少
要注意：不同教材对 discounted objective 是否额外带 $\gamma^t$ 有不同约定，必须以题目给出的 estimator 为准；Assignment 2 明确使用上面的相对折扣 return。

> [!example] 具体计算：反向得到每个动作的 reward-to-go
> 一条三步 reward 序列为 $(1,2,4)$，折扣因子 $\gamma=0.9$。从 episode 末尾开始反向递推：
>
> $$
> G_2=4,
> \qquad
> G_1=2+0.9\times4=5.6,
> $$
>
> $$
> G_0=1+0.9\times5.6=6.04.
> $$
>
> 时刻 $1$ 的动作使用 $G_1=5.6$，不会再被动作发生前的奖励 $1$ 加权。这既符合因果方向，也避免把无关的过去奖励加入梯度噪声。



### 5.2 REINFORCE：Monte Carlo policy gradient

*首次完整讲解：Lecture 5 §5.2「REINFORCE」。*

->前面推导了半天，现在终于把那些公式变成一个真正可以运行的算法——REINFORCE。

REINFORCE 是 Monte Carlo policy gradient 算法：用实际采样的完整 reward-to-go $G_t$ 估计 $Q^{\pi_\theta}(s_t,a_t)$。单步参数更新为：

$$
\Delta\theta_t
=
\alpha
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t.
$$
把它拆开：
$$\underbrace{\nabla_\theta\log\pi_\theta(a_t|s_t)} _{\text{score：提高这个动作概率的方向}} \times \underbrace{G_t}_{\text{这次动作之后实际有多好}}$$​​​

这个更新就是根据采样结果不断的调整参数sita，使得让J数值更大的动作概率更高一点


它把“当前策略生成数据”和“return 加权的 score update”接成一个循环。批量版本的一轮按以下顺序运行：

1. **初始化或读取当前策略。** 本轮开始时固定参数 $\theta_k$；这一批数据都由 $\pi_{\theta_k}$ 产生。
2. **采样完整 episodes。** 保存每个时间步的 $(s_t,a_t,r_t)$，直到终止，不能只保留总奖励。
3. **反向计算 reward-to-go。** 对每条 episode 从 $G_T=0$ 开始，用 $G_t=r_t+\gamma G_{t+1}$ 得到每个动作自己的权重。
4. **形成批量梯度估计。** 对 $m$ 条轨迹计算

   $$
   \hat g_k
   =\frac1m\sum_{i=1}^m\sum_{t=0}^{T_i-1}
   \nabla_\theta\log\pi_{\theta_k}
   (a_t^{(i)}\mid s_t^{(i)})G_t^{(i)}.
   $$

5. **更新并进入下一轮。** 令 $\theta_{k+1}=\theta_k+\alpha\hat g_k$，下一批 episode 必须由更新后的 $\pi_{\theta_{k+1}}$ 采样。

一次收集m条轨迹，对多个 episode 求平均。

$i=\text{第几条 episode}$ 
$t=\text{episode 中第几个时间步}$ 
$k=\text{第几轮参数更新}$


> [!summary] REINFORCE 数据流
> 当前策略 $\pi_{\theta_k}$ → 完整 episodes → 每个时间步的 $G_t$ → score-return 梯度 $\hat g_k$ → 新策略 $\pi_{\theta_{k+1}}$。

> [!example] 具体计算：REINFORCE 的一次随机更新
> 沿用 §2.4 的两动作策略。当前 $\theta_0=0$，本次采到 $R$，因此 $G_0=4$，而 $R$ 的 score 为 $0.5$。若 $\alpha=0.1$，则
>
> $$
> \hat g_0=0.5\times4=2,
> \qquad
> \theta_1=0+0.1\times2=0.2.
> $$
>
> 更新后
>
> $$
> \pi_{\theta_1}(R)=\sigma(0.2)\approx0.550.
> $$
>
> 下一轮采样 $R$ 的概率从 $0.5$ 上升到约 $0.55$。这一次 sampled gradient 是 $2$，并不等于真实梯度 $0.75$；若本次采到 $L$，单次更新甚至可能朝相反方向移动。REINFORCE 依靠多次采样，使平均更新逼近期望梯度。

REINFORCE 不要求 reward 可导，也不要求知道 [dynamics](academic-term-lookup:dynamics)，但必须等待足够长的 rollout 才能得到 $G_t$，并且 Monte Carlo return 的方差可能很高。



### 5.3 梯度估计量的目标：无偏、低方差且样本高效

最朴素的 score-function estimator 在课程假设下无偏，但非常 noisy。
同一个策略采样出的轨迹可能有截然不同的 return，使梯度方向在 batch 之间剧烈变化。

评价 policy gradient 算法时，需要同时问：

- estimator (比如上面的g) 的期望是否对应目标梯度；
- [variance](academic-term-lookup:variance) 是否小到能稳定学习；(它衡量不同 batch 算出来的梯度相差有多大)
- 需要多少环境 time steps 才能到达好策略；
- 每批数据能复用多少次；
- 最终只保证局部最优还是有更强保证。

本讲的两条主要降方差路径是：利用 temporal structure，以及减去 baseline。
Lecture 6 再加入 value function approximation 与 bootstrapping，它们通常用一定 bias 换取更低 variance。

### 5.4 Baseline：降方差但不改变梯度期望

*首次完整讲解：Lecture 5 §5.4「Baseline」。Lecture 6 §2.1--§2.2 将补充完整方差分析与最优 baseline 的作用域。*

**基线（baseline）** $b(s_t)$ 是一个只依赖当前状态、**不依赖本次采样动作 $a_t$** 的标量参照。更新改为：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\bigl(G_t-b(s_t)\bigr)
\right].
$$

和之前比起来，多了一个-b

因为如果做了这个动作，Gt=100，但假设其实在状态st无论做什么，通常都有98分，所以其实这个动作并没有带来多大的效益。这时候要是乘100，则会对sita造成较大改动，增大不同组数据的方差。

于是改成：Gt−b(st)



它改变单个样本的权重，却不改变期望梯度，因为对固定状态 $s$：

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=b(s)\sum_a\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1\\
&=0.
\end{aligned}
$$

所以任意 [action-independent baseline](academic-term-lookup:action-independent%20baseline) 都不引入 bias。
这里还假设策略可微、求导与动作求和可以交换，并且采样动作位于策略支持集内；若 $b$ 依赖当前采样动作，上述求和一般不再为 0，不能直接沿用证明。

[!important]
>	这个等于0的结论上面也证明过，还是很重要的

最标准的结论是：

$$\boxed{ \mathbb E_{x\sim p_\theta} \left[ \nabla_\theta\log p_\theta(x) \right] =0 }$$​(x 服从参数为$\theta$的概率分布 $p_\theta$)


在策略里就是固定状态 s 后：
$$\boxed{ \mathbb E_{a\sim\pi_\theta(\cdot|s)} \left[ \nabla_\theta\log\pi_\theta(a|s) \right] =0 }$$​

这里的
$$\nabla_\theta\log\pi_\theta(a|s)$$

就是 score。

本质上就是：

> **概率总和永远是 1，所以你怎么调整参数，各个动作概率的变化量加起来必须是 0。**


这个结论最重要的前提：按同一个分布取期望！


score 是：$\nabla_\theta\log\pi_\theta(a|s)$

那么期望必须也是：$a\sim\pi_\theta(\cdot|s)$

也就是：$\mathbb E_{a\sim\pi_\theta} [\nabla\log\pi_\theta] =0$

如果数据是另一个策略 μ 采出来的：$a\sim\mu(\cdot|s)$

一般来说：

$\boxed{ \mathbb E_{a\sim\mu} [ \nabla\log\pi_\theta(a|s) ] \neq0 }$

这也是为什么 off-policy Policy Gradient 不能直接照搬 on-policy 的很多证明，通常需要 importance sampling 等修正。

---

一个实用选择是 $b(s)\approx V^{\pi_\theta}(s)=\mathbb E[G_t\mid s_t=s]$(最常用的 baseline 是 $V^\pi(s_t)$，也就是在状态 st 下，按照当前策略继续走，平均能拿多少 return）。

此时：

- $G_t>b(s_t)$：这个动作的结果比状态下的通常表现好，提高其 log-probability；
- $G_t<b(s_t)$：结果比通常表现差，降低其 log-probability；
- $G_t\approx b(s_t)$：不需要强烈更新。

> [!example] 具体计算：baseline 如何消除本例中的采样方差
> 沿用一步两动作策略。在 $\theta=0$ 时，状态价值为
>
> $$
> b(s)=V^\pi(s)
> =0.5\times4+0.5\times1
> =2.5.
> $$
>
> 若采到 $R$，梯度样本为
>
> $$
> 0.5(4-2.5)=0.75;
> $$
>
> 若采到 $L$，梯度样本为
>
> $$
> (-0.5)(1-2.5)=0.75.
> $$
>
> 两种采样结果给出相同梯度，方差在这个一步确定性例子中降为 0，同时仍等于 §3.2 算出的真实梯度 $0.75$。一般 MDP 不会恰好达到零方差，但合适的 baseline 仍能让样本权重更集中。

> [!important] Baseline 改变的是参照系，不是任务
> 减去 $b(s_t)$ 不会修改环境奖励，也不会把优化目标换成另一个任务。它只把动作权重从“得到多少 return”改写成“比这个状态下通常表现好多少”，让梯度样本更居中。无偏性的关键条件是 baseline 不依赖本次采样动作。



## 6. Assignment Readiness：Assignment 2

### 6.1 当前已经覆盖的前置知识

- Assignment 2 §2.1：policy gradient theorem、REINFORCE 和 sampled return $G_t$。
- Assignment 2 §2.2：state baseline、无偏性条件和 baseline MSE regression。
- Assignment 2 §2.3：advantage $\hat A_t=G_t-b_\phi(s_t)$；advantage normalization 的题面解释已可读。
- 离散动作 categorical policy 与连续动作 diagonal Gaussian policy 的数学含义。

现在可以开始 Assignment 2 的基础 REINFORCE 与 baseline 部分，但这不表示已经具备整份作业的熟练度。

### 6.2 与 starter code 的精确映射

| 数学对象 | Starter code | 输入 / 输出形状与要点 |
|---|---|---|
| $\pi_\theta(\cdot\mid s)$ | `policy.py: BasePolicy.act` | observations 为 `[batch, obs_dim]`；先建 distribution，再 sample，并可返回每个样本的 log-probability |
| categorical $\pi_\theta$ | `CategoricalPolicy.action_distribution` | network 输出 `[batch, action_dim]` logits；distribution 的 log-probability 为 `[batch]` |
| diagonal Gaussian $\pi_\theta$ | `GaussianPolicy` | network 输出均值 `[batch, action_dim]`；`log_std` 为 `[action_dim]`；动作维度须作为 event 维聚合，log-probability 为 `[batch]` |
| $G_t$ | `PolicyGradient.get_returns` | 每条 path 内反向递推，最后 flatten 为 `[batch]`；不能跨 episode 传播 return |
| $\hat A_t$ | `PolicyGradient.calculate_advantage` | 无 baseline 时等于 returns；有 baseline 时调用 `BaselineNetwork.calculate_advantage` |
| advantage normalization | `PolicyGradient.normalize_advantage` | 输入输出均为 `[batch]`；按 batch 做零均值、单位标准差标准化 |
| $\log\pi_\theta(a_t\mid s_t)\hat A_t$ | `PolicyGradient.update_policy` | observations `[batch, obs_dim]`；离散 actions `[batch]`，连续 actions `[batch, action_dim]`；optimizer 最小化负 objective |
| $b_\phi(s)$ | `BaselineNetwork.forward` | network 原输出 `[batch, 1]`，squeeze 后必须是 `[batch]` |
| $G_t-b_\phi(s_t)$ | `BaselineNetwork.calculate_advantage` | returns 与 baseline prediction 都是 `[batch]`；输出 NumPy `[batch]` |
| $\lVert b_\phi(s_t)-G_t\rVert^2$ | `BaselineNetwork.update_baseline` | 对 `[batch]` prediction 和 return 做 MSE，再 backward 和 optimizer step |

实现时最容易错的两点：连续动作的 `Normal.log_prob` 若仍返回 `[batch, action_dim]`，必须把动作维变成 event 维；以及 PyTorch optimizer 做 gradient descent，所以 policy objective 前要取负号。

课件中的 batch estimator 以 trajectory 数量 $m$ 归一化，而 Assignment 2 的 objective 对 flatten 后的全部 time steps 取 mean，即除以 $\sum_i T_i$。两者使用相同的 score-return 机制，但在 episode 长度不同时会产生不同的整体缩放和 trajectory 权重；实现作业时应服从题面与 starter code 的 flattened-batch 约定。

### 6.3 尚未覆盖与 mastery 状态

- PPO clipping、old-policy ratio 和 off-policy data reuse 需要后续 lecture；本讲只覆盖作业中的 REINFORCE / baseline 基础。
- Advantage normalization 已能按 Assignment 2 的定义实现，但课程层面的更完整估计方法还会继续出现。
- 当前没有记录 quiz、独立推导或实现通过测试的 mastery evidence；“笔记覆盖完成”不等于“已经掌握”。

推荐下一步：先不看答案，手推 §9 自测题 2、4、6，再实现 `get_returns` 和基础 `update_policy`，用小数组检查 episode 边界和形状。

## 7. 本讲必会公式

1. 轨迹目标（首次推导：§3.1）：

   $$
   J(\theta)=\sum_\tau P(\tau;\theta)R(\tau).
   $$

2. 似然比恒等式（likelihood-ratio identity，首次推导：§3.2）：

   $$
   \nabla_\theta J(\theta)
   =
   \mathbb E_\tau
   \left[R(\tau)\nabla_\theta\log P(\tau;\theta)\right].
   $$

3. 轨迹 score 分解（首次推导：§3.3）：

   $$
   \nabla_\theta\log P(\tau;\theta)
   =
   \sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
   $$

4. Softmax score（首次推导：§4.1）：

   $$
   \nabla_\theta\log\pi_\theta(a\mid s)
   =
   \phi(s,a)-\mathbb E_{a'\sim\pi_\theta}[\phi(s,a')].
   $$

5. Gaussian mean score（首次推导：§4.2）：

   $$
   \nabla_\theta\log\pi_\theta(a\mid s)
   =
   \frac{a-\mu_\theta(s)}{\sigma^2}\phi(s).
   $$

6. REINFORCE（首次推导：§5.1--5.2）：

   $$
   \hat g
   =
   \frac1m\sum_{i=1}^m\sum_t
   \nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})G_t^{(i)}.
   $$

7. Baseline estimator（首次推导：§5.4）：

   $$
   \hat g_b
   =
   \frac1m\sum_{i=1}^m\sum_t
   \nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
   \left(G_t^{(i)}-b(s_t^{(i)})\right).
   $$

## 8. 容易混淆点

1. **Policy gradient vs gradient descent**：目标是最大化 return，因此数学上做 ascent；代码常通过最小化负 objective 实现。
2. **$P(\tau;\theta)$ vs $\pi_\theta(a\mid s)$**：前者是整条轨迹概率，后者只是一个状态下的单步动作概率。
3. **不需要 dynamics model vs 不需要环境**：不需要知道转移公式或对它求导，但仍需 rollout 数据。
4. **Return sample vs $Q^\pi$**：$G_t$ 是随机样本；$Q^\pi(s_t,a_t)$ 是条件期望。
5. **Baseline vs reward shaping**：baseline 只改变梯度 estimator 的中心，不修改环境 reward 或优化目标。
6. **Baseline 的无偏条件**：$b(s_t)$ 不能依赖本次采样的 $a_t$；否则标准证明不成立。
7. **真实梯度 vs sampled gradient**：$\nabla_\theta J(\theta)$ 是期望；单个 $R(\tau)\nabla\log P(\tau;\theta)$ 是随机样本，甚至可能暂时指向相反方向。
8. **Lec5 post vs pre**：课堂 post deck 在第 61 页的 baseline 引入处停止；pre-only 的 vanilla PG、actor-critic 和 N-step 内容不计入本讲覆盖，在 Lecture 6 重新按课堂来源讲解。

## 9. 自测题

### 题目

1. 为什么 Aliased Gridworld 中随机策略可能优于所有可表示的确定性策略？
2. 从 $J(\theta)=\sum_\tau P(\tau;\theta)R(\tau)$ 推到 likelihood-ratio 形式，中间使用了什么恒等式？
3. 为什么轨迹 score 中的环境转移项会消失？这个结论依赖什么假设？
4. 已知 softmax policy 在某状态下两动作概率为 $[0.8,0.2]$，动作 one-hot 特征分别为 $[1,0]$ 和 $[0,1]$。实际选中第二个动作时，score 是多少？
5. Reward-to-go 为什么比把整条 $R(\tau)$ 乘给每个时间步更合理？
6. 证明任意 action-independent baseline 不改变期望 policy gradient。
7. 为什么 $G_t-b(s_t)$ 仍是随机的梯度权重，而不是一个真实价值函数？
8. Assignment 2 的连续动作 policy 为什么不能让 `log_prob` 保持 `[batch, action_dim]`？

<details>
<summary>查看答案</summary>

1. 两个真实状态被映射成相同表示，确定性策略必须在两处采取同一方向，可能被困；随机策略能在相同表示下交替尝试两个方向并最终离开。
2. 使用 $\nabla p=p\nabla\log p$，也就是 $\nabla p/p=\nabla\log p$。
3. $\mu(s_0)$ 和 $P(s_{t+1}\mid s_t,a_t)$ 不依赖 policy 参数 $\theta$，所以对 $\theta$ 的梯度为 0。若 dynamics 也由 $\theta$ 参数化，则不能删除。
4. 选中第二个动作的特征是 $[0,1]$，期望特征为 $0.8[1,0]+0.2[0,1]=[0.8,0.2]$，所以 score 为 $[-0.8,0.8]$。
5. 时刻 $t$ 的动作不能影响此前奖励；保留此前奖励只会加入期望为 0 的噪声。Reward-to-go 保留从 $t$ 开始的因果后果。
6. 对固定 $s$，$\mathbb E_a[b(s)\nabla\log\pi(a\mid s)]=b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla1=0$。
7. $G_t$ 来自一条 sampled trajectory，$b(s_t)$ 通常也是学习得到的近似；两者之差仍随轨迹和估计误差变化，只是比未中心化的 $G_t$ 更适合作为低方差权重。
8. 每个状态动作对在 policy objective 中需要一个标量 log-probability。Diagonal Gaussian 的联合 log-probability 是各动作维 log-probability 之和，因此输出应为 `[batch]`。

</details>

## 10. 本讲小结

Lecture 5 把 RL control 从“拟合价值后选动作”转成“直接优化动作分布”。似然比恒等式（likelihood-ratio identity）让梯度作用在 policy 的 log-probability 上；轨迹分解使未知 dynamics 从梯度中消失；temporal structure 再把每个动作的权重缩小为它自己的 reward-to-go，由此得到 REINFORCE。

REINFORCE 数学直接但方差高。减去 action-independent baseline 不改变期望梯度；使用 $V^\pi(s)$ 的近似作为参照，能把动作结果解释成“高于或低于该状态的通常表现”。至于怎样训练这一参照、怎样用 critic 或 bootstrapped target 替代 Monte Carlo return，属于 Lecture 6。

## 11. 延伸阅读

### 11.1 经典基础

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 13：policy gradient、REINFORCE 与 baseline；也是课件指定阅读。
- Williams (1992), *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*：REINFORCE 的经典来源。

### 11.2 前沿动态

截至 2026-07-29 核验：本讲不单列额外前沿项目。Policy gradient 的现代应用会在 PPO、RLHF 和后续 actor-critic 内容出现时结合对应一手资料展开，避免在 REINFORCE 基础讲解中提前堆叠结论。
