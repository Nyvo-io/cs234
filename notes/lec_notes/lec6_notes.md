---
title: CS234 Lecture 6 - Policy Gradient II and PPO
aliases:
  - CS234 Lec6
tags:
  - cs234
  - reinforcement-learning
  - policy-gradient
  - ppo
---

# CS234 Lecture 6 Notes: Policy Gradient II and PPO

来源：`lecture/lecture6post.pdf`，CS234 Winter 2026，Emma Brunskill。本文以课堂 post deck 为准，共 48 个物理 PDF 页面；pre deck 共 73 页，其中第 49--73 页没有出现在 post deck，不计入本讲课堂覆盖。

笔记规范：cs234-rl-tutor v2。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-07-28（延伸阅读采用课件明确列出的经典论文；本讲不加入时效性前沿结论）。

## 0. 本讲覆盖清单

- [x] 第 1--8 页：policy gradient 回顾、本讲目标与课程结构。
- [x] 第 9--18 页：REINFORCE、state baseline 的无偏性、方差最小化视角、vanilla policy gradient 与 value baseline。
- [x] 第 19--23 页：Monte Carlo target 的替代方案、actor-critic 与 advantage 形式；重复概念回引 Lecture 5。
- [x] 第 24--31 页：vanilla policy gradient 的样本效率、步长选择与参数空间距离问题。
- [x] 第 32--40 页：performance difference lemma、discounted state distribution、单步 importance ratio、surrogate objective 与 KL 误差界。
- [x] 第 41--45 页：PPO 的 adaptive KL penalty 与 clipped objective。
- [x] 第 46--48 页：正负 advantage 下 clipped objective 的分段行为。
- [x] 映射 Assignment 2 的 `PPO.update_policy`、旧 log-probability 缓存与书面题。
- [x] 来源补充：`lecture/lecture5pre.pdf` 第 72--75 页的 N-step target 单列为 §2.5，不计入 Lecture 6 post deck 的课堂覆盖。

**视觉材料决策**：第 30 页的 performance collapse、第 31 页的 parameter--policy mismatch、第 48 页的 PPO 正负 advantage 曲线直接嵌入；baseline 推导、performance difference lemma 和算法伪代码改写为可搜索的 Markdown。第 49--73 页仅在 pre deck 中出现，不作为 post deck 的课堂内容补入。

## 1. 本讲主线

Lecture 5 已经得到一个可用但昂贵的算法：REINFORCE 用当前策略采样完整轨迹，再用 reward-to-go 更新策略。Lecture 6 要解决两个实际问题：**梯度太 noisy，以及每批 on-policy 数据只用一次太浪费。**

用采样出来的平均值去估算梯度，方差较大，很不稳定，梯度太noisy。

当前策略是：πθ

机器人按照它行动：$a_t\sim \pi_\theta(\cdot|s_t)$

得到：$(s_t,a_t,r_t,s_{t+1})$

然后你还是拿这些数据去优化这个 $\pi_\theta$。

这就是：$\boxed{\text{behavior policy}=\text{target policy}}$：on policy

而off-policy：A 策略采数据，B 策略拿来学

```
昨天的策略 π_old
      ↓
和环境交互
      ↓
存进 replay buffer
      ↓
今天已经变成 π_new
      ↓
还在拿昨天的数据训练 π_new
```

所以说按照策略跑了 **100 条 episode**，得到了几万条：$(s_t,a_t,r_t,s_{t+1})$

这些数据采集可能非常贵。

然后你用这些数据计算梯度：$\hat g$

更新一次：$\theta_{\text{old}} \rightarrow\theta_{\text{new}}$

这时候原来数据就变成旧数据，但策略已经变成了新策略，这样就得重新采集数据

---


整讲可以压缩成五步：

1. 用 state baseline 把 return 改写成“相对该状态通常表现的增量”，在不改变期望梯度的前提下降低方差。
2. 用 value function 作为 baseline，并允许 critic 提供 $Q$ 或 advantage estimate，形成 actor-critic。
3. 认识 vanilla policy gradient 的两个困难：旧数据很快失效；参数空间中的小步不一定是策略分布中的小步。
4. 用 performance difference lemma 把新旧策略的真实性能差写成旧策略 advantage 的期望，再近似成可由旧策略数据估计的 surrogate objective。
5. PPO 用 KL penalty 或 clipped probability ratio 限制策略变化，让同一批数据可以做多个 minibatch update，同时降低灾难性大步的风险。

> [!important] 两条问题轴
> Baseline / critic 主要处理的是**梯度信号质量**，PPO 主要处理的是**数据复用与更新幅度**。二者可以同时使用，但不能把“低方差 advantage”误认为“策略步一定足够小”，也不能把 clipping 误认为“critic 已经准确”。


## 2. 从 baseline 到 actor-critic

### 2.1 Lecture 5 的重复内容只作回引

*首次完整讲解：Lecture 5 §5.2「REINFORCE」与 §5.4「Baseline」。本节只补充：Lecture 6 第 9--13 页对 baseline 无偏性的课堂推导。*

本讲第 9--13 页重复 REINFORCE 与 baseline 的基本公式。对一批当前策略轨迹，带 state baseline 的 estimator 是：

$$
\hat g
=
\frac{1}{m}
\sum_{i=1}^m\sum_{t=0}^{T_i-1}
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\left(G_t^{(i)}-b(s_t^{(i)})\right).
$$

Baseline 不引入 bias 的核心仍是：对固定 $s$，若 $b(s)$ 不依赖本次采样动作，则

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1\\
&=0.
\end{aligned}
$$

本讲的新内容不是再次证明这个等式，而是回答：**什么 baseline 最能降低方差，为什么 $V^\pi(s)$ 是实用选择？**

### 2.2 固定状态下，单个梯度项的最优 baseline

*首次完整讲解：Lecture 6 §2.2「固定状态下，单个梯度项的最优 baseline」。*

课件第 14--15 页先聚焦一个时间步，而不是直接优化整条 trajectory 中所有梯度项之和。固定状态 $s$，记该时间步的 policy score 为向量

$$
z=\nabla_\theta\log\pi_\theta(A_t\mid s),
$$

对应的随机梯度项是 $z(G_t-b)$。因为 action-independent baseline 不改变这一项的期望，所以最小化其方差等价于最小化二阶矩：

$$
b_{\mathrm{term}}^*(s)
=
\arg\min_b
\mathbb E\left[\lVert z\rVert_2^2(G_t-b)^2\mid s_t=s\right].
$$

对标量 $b$ 求导并令其为 0：

$$
\begin{aligned}
0
&=
\mathbb E\left[2\lVert z\rVert_2^2(b-G_t)\mid s_t=s\right],\\
b_{\mathrm{term}}^*(s)
&=
\frac{
\mathbb E[\lVert z\rVert_2^2G_t\mid s_t=s]
}{
\mathbb E[\lVert z\rVert_2^2\mid s_t=s]
}.
\end{aligned}
$$

这是用 score norm 平方加权的 return 条件均值。若不同动作样本的 $\lVert z\rVert_2^2$ 变化不大，就近似为普通条件均值：

$$
b_{\mathrm{term}}^*(s)
\approx\mathbb E[G_t\mid s_t=s]
=V^{\pi_\theta}(s).
$$

**自拟数值例子**：同一状态下有两类样本，$(G,\lVert z\rVert^2)=(2,1)$ 与 $(8,3)$，各占一半。普通 return 均值是 $5$，而方差最优常数 baseline 是

$$
b_{\mathrm{term}}^*=
\frac{0.5\times1\times2+0.5\times3\times8}
{0.5\times1+0.5\times3}
=\frac{13}{2}=6.5.
$$

高 score-norm 样本对这一梯度项的波动影响更大，所以最优 baseline 更靠近它的 return。实践中直接估计这个加权解不一定划算，拟合 $V^\pi(s)$ 是更简单且通常有效的近似。

> [!warning] 最优性的作用域
> 上式精确优化的是“固定状态下一个 score-return 梯度项”的条件方差。整条 trajectory 的 estimator 是多个时间步之和，时间步之间可能有协方差；不额外假设这些交叉项可忽略时，不能把 $b_{\mathrm{term}}^*(s)$ 宣称为整个轨迹梯度估计器的一般全局最优 baseline。

> [!important] “任意 baseline 无偏”不等于“任意 baseline 都降方差”
> Action-independent 只保证减去 baseline 后期望梯度不变。若 baseline 与 return 毫无关系，方差可能几乎不降，甚至更大；$V^\pi(s)$ 的价值来自它接近条件期望，而不是来自“baseline”这个名字。

### 2.3 Vanilla policy gradient：把 policy 与 baseline 接成训练循环

*首次完整讲解：Lecture 6 §2.3「Vanilla policy gradient：把 policy 与 baseline 接成训练循环」。*

Vanilla policy gradient（VPG）把 Lecture 5 的 REINFORCE 与一个可学习的 state baseline 组合起来。它每轮解决两个不同的学习问题：baseline 参数 $\phi$ 拟合 return，policy 参数 $\theta$ 最大化 baseline-centered policy objective。

一轮训练按以下顺序运行：

1. **固定当前策略并收集数据。** 用 $\pi_{\theta_k}$ 采样一批完整 trajectories。
2. **计算 Monte Carlo targets。** 对每条 trajectory 反向计算所有 $G_t$。
3. **冻结本轮 policy 权重。** 用更新前的 baseline 计算

   $$
   \hat A_t=G_t-b_{\phi_k}(s_t).
   $$

4. **训练 baseline。** 最小化

   $$
   \mathcal L_V(\phi)
   =\frac1N\sum_{t=1}^{N}
   \left(b_\phi(s_t)-G_t\right)^2,
   $$

   让 $b_\phi$ 更接近当前策略的 state value。
5. **更新 policy。** 使用已经冻结的 $\hat A_t$ 形成

   $$
   \hat g_k
   =\frac1N\sum_{t=1}^{N}
   \nabla_\theta\log\pi_{\theta_k}(a_t\mid s_t)\hat A_t,
   \qquad
   \theta_{k+1}=\theta_k+\alpha\hat g_k.
   $$

> [!summary] VPG 数据流
> 当前 policy → trajectories → returns $G_t$ → 冻结的 $\hat A_t$；同一批数据随后分成两路：$G_t$ 监督 baseline，$\hat A_t$ 加权 policy score。

**走完一批更新。** 继续使用 Lecture 5 的一步两动作例子。令 $\theta_0=0$，batch 中恰有一次 $R$ 和一次 $L$，对应 return 为 $4$ 和 $1$；当前 baseline 为 $2.5$。于是

$$
\hat A_R=4-2.5=1.5,
\qquad
\hat A_L=1-2.5=-1.5.
$$

两个动作的 score 分别为 $0.5$ 与 $-0.5$，所以 batch gradient 是

$$
\hat g_0
=\frac12\left[0.5\times1.5+(-0.5)\times(-1.5)\right]
=0.75.
$$

若 $\alpha=0.1$，则 $\theta_1=0.075$，下一轮 $R$ 的概率变为 $\sigma(0.075)\approx0.519$。同时，以 returns $\{4,1\}$ 做常数回归时，$2.5$ 已是本 batch 的 MSE 最优 baseline。下一轮必须重新采样，因为 policy 和它诱导的数据分布已经改变。

实现中应把 $\hat A_t$ 视为 policy update 的常数；否则 actor loss 可能沿 baseline 或 target 的计算图产生并非该 estimator 所定义的额外梯度。

### 2.4 Advantage 与 actor-critic：替换 Monte Carlo target

*首次完整讲解：Lecture 6 §2.4「Advantage 与 actor-critic：替换 Monte Carlo target」。*

优势函数（advantage function）是状态动作标量函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

它回答“先做动作 $a$，比在状态 $s$ 按当前策略平均行动好多少”。例如 $Q^\pi(s,L)=7$、$Q^\pi(s,R)=3$，策略以相同概率选择两动作，则

$$
V^\pi(s)=0.5\times7+0.5\times3=5,
$$

$$
A^\pi(s,L)=2,
\qquad
A^\pi(s,R)=-2.
$$

正 advantage 表示该动作优于策略平均，负 advantage 表示更差；它不是动作的绝对长期回报。

Actor-critic（演员--评论家）用两个可学习对象把这一关系落地：

- **actor** $\pi_\theta(a\mid s)$ 输出动作分布并与环境交互；
- **critic** $\hat V_\phi(s)$ 或 $\hat Q_\phi(s,a)$ 估计当前 actor 的价值，为 policy update 提供 target 或 baseline。

>	REINFORCE 原来让 actor 直接拿整条轨迹的 Gt​ 判断“这个动作好不好”，等这一条episode走完，他是monte carlo的，不顾及下一步价值，不bootstrap；
>	Actor-Critic 改成再训练一个 critic，专门帮 actor 判断“刚才这个动作比预期好还是差，不一定要走完。
>	
>	这两个的核心区别：如何估计当前动作到底有多好。


actor 负责：

> **在这个状态下，我该选什么动作？**

critic 负责：

> **这个状态大概有多好？**



课件以 A3C（Mnih et al., 2016）作为 actor-critic 的代表方法，但本讲只使用它说明这一算法家族，并不展开 A3C 的异步训练机制。

Lecture 6 第 21--23 页的变化是：不再强制用完整 $G_t$ 估计 $Q^\pi$，而允许 critic 产生更低方差但可能有 bias 的 $\hat Q_t$，再构造

$$
\hat A_t=\hat Q_t-\hat V_\phi(s_t),
$$

$$
\hat g
=\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t.
$$


最简单的一步 actor-critic 是使用 bootstrap target 去计算Qt

$$
y_t=r_t+\gamma\hat V_\phi(s_{t+1}),
\qquad
\hat A_t=y_t-\hat V_\phi(s_t).
$$

若 $s_{t+1}$ 是 terminal state，则约定 $\hat V_\phi(s_{t+1})=0$，所以 target 退化为 $y_t=r_t$，不能从 episode 之外继续 bootstrap。

**走完一个 transition。** 若 $r_t=1$、$\gamma=0.9$、$\hat V_\phi(s_t)=2$、$\hat V_\phi(s_{t+1})=3$，则

$$
y_t=1+0.9\times3=3.7,
\qquad
\hat A_t=3.7-2=1.7.
$$

Critic 用 $3.7$ 作为 target，把 $\hat V_\phi(s_t)$ 从 $2$ 往上调整；actor 则用 $1.7$ 加权已选动作的 score。若该 score 在某个参数方向上的分量为 $0.4$、actor 学习率为 $0.1$，该参数本次增加 $0.1\times0.4\times1.7=0.068$。更新后的 actor 负责下一批采样，更新后的 critic 负责下一轮评价。

> [!important] 四个对象不要压成一个词
> $G_t$ 是单条轨迹的随机 complete return；$Q^\pi(s_t,a_t)$ 是条件期望；$V^\pi(s_t)$ 是当前状态下对动作再平均的期望；$\hat A_t$ 是用于更新的相对优势估计。Critic 不替 actor 选动作，它只提供学习信号。

---

之前学过：$V^\pi(s) = \mathbb E_\pi[G_t|s_t=s]$

但真实的：$V^\pi(s)$，我们通常不知道。

所以训练一个神经网络：$\hat V_\phi(s)$去近似它：

$\hat V_\phi(s)\approx V^\pi(s)$


$y_t = r_t+\gamma\hat V_\phi(s_{t+1})$

就是：

> 我已经执行了at拿到了 3 分，并且来到 st+1​；critic 认为从 st+1 往后还能拿约 12 分，也就是$\hat V_\phi(s)$的值。

用我自己对下一个状态的估计，帮助估计当前状态就是bootstrap

其实这里的yt就是TD target，Actor-Critic 的 critic 部分，本质上就是 TD learning

而advantage ：

$\boxed{ \hat A_t = r_t+\gamma\hat V_\phi(s_{t+1}) -\hat V_\phi(s_t) }$

就是TD error

假如它大于0，则表示刚才选的 at​，比我原本对这个状态的平均预期更好，所以actor要提高这个动作发生的概率，这就是advantage的作用


### 2.5 来源补充：N-step target 的 bias--variance 权衡

*首次完整讲解：Lecture 6 §2.5「来源补充：N-step target 的 bias--variance 权衡」。本节取自 `lecture/lecture5pre.pdf` 第 72--75 页，不计入 Lecture 6 post deck 的课堂覆盖。*

N-step target 在完整 Monte Carlo return 与一步 bootstrap 之间选择折中，不再像上面只走一步就开始判断，让critic猜。现在是再走n步，取得真实的n步数据，n个r，再让critic去猜。
若从 $t$ 起至少还有 $n$ 个非终止 transition，则

$$
\hat Q_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n\hat V_\phi(s_{t+n}),
\qquad
\hat A_t^{(n)}=\hat Q_t^{(n)}-\hat V_\phi(s_t).
$$

若在 $t+n$ 前已经终止，就只累加终止前的奖励，不添加 terminal 之后的 bootstrap 项。比如 $n=2$、$r_t=1$、$r_{t+1}=2$、$\gamma=0.9$、$\hat V_\phi(s_{t+2})=4$，则

$$
\hat Q_t^{(2)}
=1+0.9\times2+0.9^2\times4
=6.04.
$$

若 $\hat V_\phi(s_t)=2$，相应 advantage estimate 为 $4.04$。

$n=1$ 较早使用 critic，通常 variance 较低但更受 value bias 影响(随机 reward 用得少，所以通常比较稳定,但非常依赖critic对下个状态的猜测，bias风险较大)；

直到 episode 结束的 Monte Carlo target 使用更多真实奖励，通常 variance 较高但没有 bootstrap approximation，bias会小。 真实 reward用的多，不代表会稳定，因为环境和策略都有随机性。而 critic 相当于提前告诉你：$\hat V(s) \approx \text{从这里出发很多种未来的平均结果}$。所以更平滑



## 3. [Vanilla policy gradient](academic-term-lookup:vanilla%20policy%20gradient) 为什么难以稳定复用数据

### 3.1 On-policy expectation 与样本效率

*首次完整讲解：Lecture 6 §3.1「On-policy expectation 与样本效率」。*

Vanilla policy gradient 的期望是对当前策略 $\pi_{\theta_k}$ 的轨迹分布计算的：

$$
\nabla_\theta J(\theta_k)
=
\mathbb E_{\tau\sim\pi_{\theta_k}}
\left[
\sum_t\gamma^t
\nabla_\theta\log\pi_{\theta_k}(a_t\mid s_t)
A^{\pi_{\theta_k}}(s_t,a_t)
\right].
$$

做一次更新后，参数变成 $\theta_{k+1}$，原 batch 却来自 $\pi_{\theta_k}$。若直接把旧数据当成新策略数据反复使用，就忽略了 data distribution mismatch。最稳妥的 vanilla 做法是一次 update 后丢弃 batch，但样本效率很低。

这里的目标不是任意复用 replay buffer，而是在新旧策略足够接近时，对同一批数据做有限次更新，并显式控制 mismatch。

### 3.2 步长过大可能让性能坍塌

Policy gradient 的基本更新是

$$
\theta_{k+1}=\theta_k+\alpha_k\hat g_k.
$$

$\alpha_k$ 太小则学习慢；太大则可能越过局部高回报区域，使真实性能突然下降。更麻烦的是，合适步长会随当前参数与网络敏感度变化，固定 learning rate 很难始终合适。

![[lec6-step-size-collapse-p30.png|900]]

*图：参数轴上的一次大步越过高价值区域并落入低性能区域。来源：`lecture/lecture6post.pdf`，物理 PDF 第 30 页。*

Adam、advantage normalization 等方法能调整参数更新尺度，但没有直接定义“策略分布究竟改变了多少”，

==参数变化小不等于策略变化小==


### 3.3 参数距离不等于策略距离

*首次完整讲解：Lecture 6 §3.3「参数距离不等于策略距离」。*

我们真正关心的其实不是：
$\theta_{k+1}-\theta_k$有多大

而是：
$\boxed{ \pi_{\theta_{k+1}}(a|s) \text{ 与 } \pi_{\theta_k}(a|s) \text{ 差多少} }$


Policy space（策略空间）是允许策略的集合。对有限 tabular MDP，它是所有行随机矩阵：

$$
\Pi
=
\left\{
\pi\in\mathbb R^{|\mathcal S|\times|\mathcal A|}
\;\middle|\;
\pi(a\mid s)\ge0,
\ \sum_{a\in\mathcal A}\pi(a\mid s)=1
\ \forall s
\right\}.
$$

每个状态对应 probability simplex 中的一行；参数化策略则是映射 $\theta\mapsto\pi_\theta$，把参数空间中的一点送到 policy space 中的一组动作分布。问题在于这个映射通常不是等距的。

课件用二动作 Bernoulli policy 说明：

$$
\pi_\theta(a)=
\begin{cases}
\sigma(\theta), & a=a_1,\\
1-\sigma(\theta), & a=a_2.
\end{cases}
$$

同样大小的 $\Delta\theta$ 在 sigmoid 饱和区和中间区会造成不同的概率变化；深度网络中这种参数化敏感度更复杂。因此

$$
\lVert\theta'-\theta\rVert_2
\quad\text{小}
\quad\not\Rightarrow\quad
\pi_{\theta'}(\cdot\mid s)
\quad\text{在所有状态都接近}\quad
\pi_\theta(\cdot\mid s).
$$

所以你单纯调learning rate 根本不够

![[lec6-parameter-policy-distance-p31.png|900]]

*图：同一参数化中，不同 $\theta$ 对应的动作概率可能变化很快；参数空间步长不能直接代表 policy-space 距离。来源：`lecture/lecture6post.pdf`，物理 PDF 第 31 页。*



**自拟数值例子**：同样取 $\Delta\theta=0.5$。从 $\theta=0$ 到 $0.5$ 时，$\sigma(\theta)$ 从 $0.500$ 变为 $0.622$，概率变化约 $0.122$；从 $\theta=4$ 到 $4.5$ 时，它只从 $0.982$ 变为 $0.989$，概率变化约 $0.007$。同一个参数步在 policy space 中并不等距。

这推动我们在**分布空间**中度量更新，例如用 KL divergence，而不只看参数向量的欧氏距离。


## 4. 从性能差异恒等式到可优化 surrogate

### 4.1 Discounted state distribution

*首次完整讲解：Lecture 6 §4.1「Discounted state distribution」。*

本节约定在 $s_t$ 执行动作 $a_t$ 后观察 $r_t$ 与 $s_{t+1}$，所有策略共享同一个初始状态分布 $\mu$。采用 [infinite-horizon discounted objective](academic-term-lookup:infinite-horizon%20discounted%20objective)：

$$
J(\pi)
=
\mathbb E_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right],
\qquad 0\le\gamma<1.
$$
意思是用策略 π 跑轨迹，看看长期 discounted reward 的期望


state distribution：策略π访问各个状态的频率

策略诱导的 normalized discounted state distribution（归一化折扣状态分布）定义为：

$$
d^\pi(s)
=
(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t\Pr(s_t=s\mid\pi).
$$
Pr代表的是在第 t 步访问状态 s 的概率

公式表示：
>	把策略 π 在所有时间步访问状态 s 的概率，加权平均起来
>	表示如果 agent 一直按照策略 π行动，那么状态 s 在整个交互过程中，在带折扣的时间权重下，这个状态占了多少概率质量，因为越后出现的状态权重越小。
>	在rl里，G的定义就是discounted return，认为先出现状态的价值比后出现的重要的多

因为我们的 RL objective 里面每个时间步权重不一样：$1,\gamma,\gamma^2,\gamma^3,\dots$，如果前面不加1-gamma，那么单纯的Pr不满足概率分布。因为一个状态概率分布必须满足：$\sum_s d^\pi(s)=1$。

求和：

$= \sum_{t=0}^{\infty} \gamma^t \underbrace{\sum_s\Pr(s_t=s\mid\pi)}_{=1}$​

$= \sum_{t=0}^{\infty}\gamma^t$

而等比数列：$\sum_{t=0}^\infty\gamma^t = \frac{1}{1-\gamma}$



只有乘上去之后，它才是合法概率分布，因为

$$
\sum_s d^\pi(s)
=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
=1.
$$


**自拟数值例子**：$\gamma=0.5$，初始时只在 $s_0$，从 $t=1$ 起永远在 $s_1$。则

$$
d^\pi(s_0)=(1-0.5)\times1=0.5,
$$

$$
d^\pi(s_1)
=(1-0.5)(0.5+0.5^2+\cdots)
=0.5.
$$

$d^\pi$ 不是某一个固定时间步的 state distribution；它把各时间步按 $\gamma^t$ 加权后混合。因子 $(1-\gamma)$ 只负责归一化，这也是后面出现 $1/(1-\gamma)$ 的原因。



### 4.2 Performance difference lemma

*首次完整讲解：Lecture 6 §4.2「Performance difference lemma」。*

设 $\pi$ 是旧策略，$\pi'$ 是候选新策略。Performance difference lemma（策略性能差异引理）给出精确恒等式：

$$
\begin{aligned}
J(\pi')-J(\pi)
&=
\mathbb E_{\tau\sim\pi'}
\left[
\sum_{t=0}^{\infty}\gamma^t
A^\pi(s_t,a_t)
\right]\\
&=
\frac{1}{1-\gamma}
\mathbb E_{s\sim d^{\pi'},\,a\sim\pi'}
[A^\pi(s,a)].
\end{aligned}
$$

Performance Difference Lemma把“整条轨迹”重新整理成了 state-action 的形式。先抽一个状态，然后再在这个状态下抽一个动作。

精确公式：

$$\mathbb E_{ s\sim d^{\pi'}, a\sim\pi' } [A^\pi]$$

也就是：

$$\sum_s d^{\pi'}(s) \sum_a \pi'(a|s)A^\pi(s,a)$$
所以：
$$\boxed{ \underbrace{d^{\pi'}(s_t)}_{\text{前面所有动作决定“能不能走到这里”}} \; \underbrace{\pi'(a_t|s_t)}_{\text{到了这里以后当前动作怎么选}} }$$


直观理解第二个等号(也可以公式证明)：

原式是：

$\sum_{t=0}^{\infty}\gamma^t A^\pi(s_t,a_t)$

也就是沿着 π′ 的轨迹，把每个时刻的 advantage 按时间折扣后相加。

而 $d^{\pi'}(s)$已经把：

- 第 0 时刻访问状态的权重 1
- 第 1 时刻访问状态的权重 γ
- 第 2 时刻访问状态的权重 $\gamma^2$
- ……

全部压缩进一个状态分布中。

但为了让$d^{\pi'}$的概率和等于 1，定义时额外乘了 1−γ。所以从 $d^{\pi'}$ 还原原来的折扣求和时，就必须除以 1−γ。

因为：

$\sum_s d^{\pi'}(s) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t = (1-\gamma)\frac{1}{1-\gamma} =1.$


---

恒等式最重要的结构是：**用旧策略 $\pi$ 的 advantage，表达新策略 $\pi'$ 相对旧策略的真实性能变化。**
让新策略 π′ 走，但用旧策略 π 当评分标准。 但令人尴尬的是，这里是新策略，我们手里的batch数据却都是旧策略采样的，因此后面引出了 TRPO / PPO


我们最想知道的就是：J(π′)−J(π) ，如果大于0，则新策略有用。但如果真让新策略去跑很多轨迹，会花费很大。

所以我们希望利用已知的旧策略，去描述新策略好不好，于是利用旧策略的advantage
$$A^\pi(s,a) = Q^\pi(s,a)-V^\pi(s)$$
表示在状态 s 做动作 a，相比旧策略 π 正常情况下的平均表现，好多少



等式证明的核心是 Bellman telescoping

Bellman 关系告诉我们：
$$Q^\pi(s_t,a_t) = \mathbb E[ r_t+\gamma V^\pi(s_{t+1}) \mid s_t,a_t ]$$

对 $\pi'$ 的轨迹：

$$
\mathbb E[A^\pi(s_t,a_t)]
=
\mathbb E[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)].
$$

乘 $\gamma^t$ 后求和，中间的 value 项相消：也就是下面的中括号里变成了$r_0+\gamma r_1+\gamma^2r_2+\cdots - V^\pi(s_0)$

$$
\begin{aligned}
&\mathbb E_{\tau\sim\pi'}
\left[
\sum_{t=0}^{\infty}\gamma^t
(r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t))
\right]\\
&=J(\pi')-\mathbb E_{s_0\sim\mu}[V^\pi(s_0)]
=J(\pi')-J(\pi),
\end{aligned}
$$

这里的$s_0\sim \mu$不是说 s0​“属于” μ，而是：

初始状态 s0 是从初始状态分布 μ 中随机采样出来的。状态s0==服从==于初始状态分布。
写一个期望是因为s0不一定每次都相同


其中用到了新旧两策略共享 $\mu$(初始状态)，且 bounded value 与 $0\le\gamma<1$ 使尾项$\gamma^{T+1}V^\pi(s_{T+1})$消失。若起始分布也随策略改变，最后一步不能直接写成同一个 $J(\pi)$。


**自拟一状态例子**：每一步都回到同一状态，$\gamma=0.9$；动作 $a_1,a_2$ 的奖励为 1 和 3。旧策略概率为 $(0.75,0.25)$，所以每步期望奖励为 $1.5$，$J(\pi)=15$。旧策略下

$$
A^\pi(s,a_1)=-0.5,
\qquad
A^\pi(s,a_2)=1.5.
$$

若新策略概率为 $(0.25,0.75)$，则

$$
\frac{1}{1-0.9}
\mathbb E_{a\sim\pi'}[A^\pi(s,a)]
=10\left(0.25\times(-0.5)+0.75\times1.5\right)
=10.
$$

新策略每步期望奖励为 $2.5$，所以 $J(\pi')=25$，确有 $J(\pi')-J(\pi)=10$。


### 4.3 单步 importance ratio 解决动作分布 mismatch

*首次完整讲解：Lecture 6 §4.3「单步 importance ratio 解决动作分布 mismatch」。*

[Performance difference lemma](academic-term-lookup:performance%20difference%20lemma) 仍要求 $a\sim\pi'$。对固定状态，可用 [importance sampling identity](academic-term-lookup:importance%20sampling%20identity)：importance ratio就是用于概率修正的

$$
\mathbb E_{a\sim\pi'}[A^\pi(s,a)]
=
\mathbb E_{a\sim\pi}
\left[
\frac{\pi'(a\mid s)}{\pi(a\mid s)}
A^\pi(s,a)
\right],
$$

本来要求动作是必须服从新策略，但利用importance sampling ，可以把它转换成 服从于旧策略

![[Pasted image 20260805010325.png]]


前提是当 $\pi'(a\mid s)>0$ 时，$\pi(a\mid s)>0$。定义单步 probability ratio：

$$
r(s,a)=\frac{\pi'(a\mid s)}{\pi(a\mid s)}.
$$

这是单步，一个状态的一次动作采样。Performance Difference Lemma把“整条轨迹”整理成了 单个state-action 的形式，所以动作修正这块只用单步就行，但状态不可以。对于动作可以只有一个ratio，因为它只考虑状态s已经给我了，我只想把这个状态下的动作分布从 π 修正成 π′。

但如果你连状态st的分布也想精确从旧策略修正到新策略，就要考虑概率的连乘：

状态可以写成一个单独的 importance ratio：
$$\boxed{ \frac{d^{\pi'}(s)}{d^\pi(s)} }$$​​

所以从纯数学上说，完全可以写：
$$\mathbb E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)] = \mathbb E_{s\sim d^\pi,a\sim\pi} \left[ \frac{d^{\pi'}(s)}{d^\pi(s)} \frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s,a) \right]$$

但d的比值比较难算，他是全局的，按照策略pi一路跑，最终有多大权重会来到状态s。他会带来：$\pi(a_0|s_0), \pi(a_1|s_1), \ldots$

以及环境：$P(s_{t+1}|s_t,a_t)$



沿用 §4.2 的一状态例子，旧策略概率为 $(0.75,0.25)$，新策略为 $(0.25,0.75)$，所以

$$
r(s,a_1)=\frac{0.25}{0.75}=\frac13,
\qquad
r(s,a_2)=\frac{0.75}{0.25}=3.
$$

新策略把 $a_2$ 的概率提高到原来的 3 倍，因此旧策略下采到的 $a_2$ 样本在估计新策略动作期望时得到权重 3。

这个 ratio 只校正当前状态下的**动作分布**，没有校正状态来自 $d^{\pi'}$ 还是 $d^\pi$。因此精确式变成

$$
J(\pi')-J(\pi)
=
\frac{1}{1-\gamma}
\mathbb E_{s\sim d^{\pi'},\,a\sim\pi}
[r(s,a)A^\pi(s,a)],
$$

仍然不能完全用旧策略 rollouts 直接估计。因为这只修好了动作，但状态任然是新策略的

> [!important] 两层 distribution mismatch
> 新旧策略不同会同时改变“在同一状态选什么动作”和“未来会访问哪些状态”。单步 ratio $\pi'/\pi$ 精确修正第一层；把 $d^{\pi'}$ 换成 $d^\pi$ 才是在第二层做近似。只看到 ratio 就说“已经完全 off-policy 校正”是不准确的。

4.3 只解决了 action distribution mismatch

不能用 Importance Sampling 连状态也精确修掉


### 4.4 Surrogate objective 与 KL 误差界

*首次完整讲解：Lecture 6 §4.4「Surrogate objective 与 KL 误差界」。*

当 $\pi'$ 与 $\pi$ 足够接近时，近似 $d^{\pi'}\approx d^\pi$，得到 surrogate objective（代理目标）：

$$
\mathcal L_\pi(\pi')
=
\frac{1}{1-\gamma}
\mathbb E_{s\sim d^\pi,\,a\sim\pi}
\left[
\frac{\pi'(a\mid s)}{\pi(a\mid s)}
A^\pi(s,a)
\right].
$$

于是

$$
J(\pi')-J(\pi)\approx\mathcal L_\pi(\pi'),
$$
我们原先是要优化左边，现在转成优化右边

右侧可以用旧策略 $\pi$ 的数据估计。这里使用单步 ratio，而不是整条历史 probability ratio 的乘积，因此比完整轨迹 importance sampling 更不容易出现连乘导致的爆炸或消失。



沿用 §4.2 的一状态例子，旧策略是 $(0.75,0.25)$、新策略是 $(0.25,0.75)$，所以两个动作的 ratio 是 $(1/3,3)$。在旧策略下计算：

$$
\mathbb E_{a\sim\pi}[r(s,a)A^\pi(s,a)]
=0.75\times\frac13\times(-0.5)
+0.25\times3\times1.5
=1.
$$

乘 $1/(1-0.9)=10$ 后仍得到性能差 10。在一状态 MDP 中 $d^{\pi'}=d^\pi$，因此这个例子的 surrogate 恰好等于精确式；一般多状态 MDP 不具备这个边界条件。



课件给出的 relative policy performance bound 是：存在与问题有关的常数 $C$，使

$$
\left|
J(\pi')-
\left(J(\pi)+\mathcal L_\pi(\pi')\right)
\right|
\le
C
\sqrt{
\mathbb E_{s\sim d^\pi}
[D_{\mathrm{KL}}(\pi'\Vert\pi)[s]]
}.
$$

它给出 PPO 的核心动机：若新旧 policy 在 KL 意义下接近，surrogate 与真实 improvement 的误差可控。

$D_{\rm KL}(\pi'\|\pi)$

它衡量：

> **新策略的动作概率分布和旧策略到底差多远。**



### 4.5 KL divergence 度量 policy-space 变化

*首次完整讲解：Lecture 6 §4.5「KL divergence」。*

离散分布的 [Kullback--Leibler divergence](academic-term-lookup:kullback--leibler%20divergence) 是

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_xP(x)\log\frac{P(x)}{Q(x)}.
$$

对固定状态的两个 policy：

$$
D_{\mathrm{KL}}(\pi'\Vert\pi)[s]
=
\sum_a\pi'(a\mid s)
\log\frac{\pi'(a\mid s)}{\pi(a\mid s)}.
$$
[!example]
>	
>	现在先**固定一个状态 s**。
>	
>	例如这个状态有三个动作：a1,a2,a3
>	
>	旧策略：

$\pi(\cdot|s) = (0.5,0.3,0.2)$

意思是：π(a1∣s)=0.5.     π(a2∣s)=0.3.     π(a3∣s)=0.2

新策略：$\pi'(\cdot|s) = (0.6,0.25,0.15)$

那么：$D_{\mathrm{KL}}(\pi'\|\pi)[s] = \sum_a \pi'(a|s) \log \frac{\pi'(a|s)} {\pi(a|s)}$

其中：$\boxed{\sum_a}$​

就是：$a_1+a_2+a_3$

三个动作都算进去。

$\begin{aligned} D_{\mathrm{KL}}(\pi'\|\pi)[s] ={}& 0.6\log\frac{0.6}{0.5}\\ &+ 0.25\log\frac{0.25}{0.3}\\ &+ 0.15\log\frac{0.15}{0.2} \end{aligned}$

最后得到一个数。表示：

> **在状态 s 下，新旧两个动作概率分布差多大。**

---

它满足 $D_{\mathrm{KL}}(P\Vert Q)\ge0$、同分布时为 0，但通常不对称：

$$
D_{\mathrm{KL}}(P\Vert Q)
\ne
D_{\mathrm{KL}}(Q\Vert P).
$$

**自拟数值例子**：旧策略为 $(0.5,0.5)$，新策略为 $(0.6,0.4)$，则

$$
D_{\mathrm{KL}}(\pi'\Vert\pi)
=0.6\log1.2+0.4\log0.8
\approx0.0201.
$$

KL 是分布变化的度量，不是 reward difference，也不是参数向量距离。

---


到这里其实已经把 PPO 的动机完整搭好了。

我们现在想：

$$\max_{\pi'}\mathcal L_\pi(\pi')$$

也就是尽量让 surrogate 变大。

但根据刚才的 bound：

> 如果 π′ 离 π 太远，surrogate 就可能骗人，就不能采用近似

所以理想目标变成：
$$\boxed{ \text{尽量提高 }\mathcal L_\pi(\pi') }$$

同时：
$$\boxed{ \text{不要让 }\pi'\text{ 离 }\pi\text{ 太远} }$$​


## 5. Proximal Policy Optimization

前面 4.x 那么多铺垫，最后就是为了回答：

> **我想把 old policy 采到的一批数据重复训练几次，但又不能让 current policy 越跑越远，怎么办？**

### 5.1 PPO 的两种近端更新思路

*首次完整讲解：Lecture 6 §5.1「PPO 的两种近端更新思路」。*

PPO（Proximal Policy Optimization，近端策略优化）是一族近似限制相邻策略变化的方法。课件介绍两种变体：

1. Adaptive KL penalty：直接在 surrogate 上减去 KL 惩罚，并动态调整惩罚系数。
2. Clipped objective：把 probability ratio 的有利变化截在区间 $[1-\epsilon,1+\epsilon]$ 附近。

两种变体共享同一个外层训练循环：

1. **冻结 old policy。** 令 $\pi_k=\pi_{\theta_k}$，用它采集当前数据集 $D_k$。
2. **估计学习信号。** 在 $D_k$ 上计算并冻结每个时间步的 $\hat A_t^{\pi_k}$。
		一条样本现在差不多变成：$\boxed{ (s_t,a_t,\log\pi_{\rm old}(a_t|s_t),\hat A_t) }$​
3. **复用当前 batch。** 保持 old-policy 概率和 advantage 不变，对 current policy 参数做 $K$ 次 [minibatch gradient update](academic-term-lookup:minibatch%20gradient%20update)；两种 PPO 变体只在这里使用不同 [objective](academic-term-lookup:objective)。
4. **检查策略变化。** Adaptive KL 版本测量实际平均 KL 并调整惩罚系数；clipped 版本通常也监控 KL，但 clipping 本身不强制 hard threshold。
5. **刷新数据分布。** 令更新后的 policy 成为下一轮 old policy，重新与环境交互，而不是继续无限复用 $D_k$。

> [!summary] PPO 数据流
> old policy 采样 $D_k$ → 冻结 old log-probability 与 $\hat A_t$ → 同一 batch 上做有限次 current-policy 更新 → 检查变化 → 刷新 old policy 并重新采样。


```
old policy 采一批数据
      ↓
冻结 old probability 和 advantage
      ↓
current policy update
      ↓
还是这一批数据
      ↓
再 update
      ↓
再 update
      ↓
有限 K 次
      ↓
数据扔掉
      ↓
用新 policy 重新采
```


Adaptive KL penalty 写为：

$$
\theta_{k+1}
=
\arg\max_\theta
\left[
\mathcal L_{\theta_k}(\theta)
-\beta_k\bar D_{\mathrm{KL},k}(\theta)
\right],
$$

为避免只看参数下标而误判 KL 方向，把课件使用的平均 KL 明确写成

$$
\bar D_{\mathrm{KL},k}(\theta)
=
\mathbb E_{s\sim d^{\pi_k}}
\left[
D_{\mathrm{KL}}
(\pi_{\theta_k}(\cdot\mid s)\Vert\pi_\theta(\cdot\mid s))
\right].
$$

第一项：$L_{\theta_k}(\theta)$ 就是 surrogate objective， 我们希望它越大越好，但也不能最大化，但current policy会跑太远，导致不能取约等于，所以-$\beta_k\bar D_{\mathrm{KL}}$ (为什么可以这么写，在lec7有理论证明)

KL在这就是罚款，β是罚款力度，Adaptive表示β不是永远固定


若 target KL 为 $\delta$ 要求两个策略的差距控制在 $\delta$ 左右，课件规则是：实际 KL 大于 $1.5\delta$ 时令 $\beta_{k+1}=2\beta_k$， 加大处罚力度 ， 会使得优化器努力调小Dkl的值 ；小于 $\delta/1.5$ 时令 $\beta_{k+1}=\beta_k/2$；否则保持不变。

例如 $\delta=0.01$、$\beta_k=0.5$，本轮用 $D_k$ 做完 $K$ 次 minibatch update 后测得平均 KL 为 $0.018$。因为 $0.018>1.5\times0.01$，下一轮使用 $\beta_{k+1}=1.0$，对偏离 old policy 的惩罚变强。

参数更新已经发生，算法不会撤销这一轮；因此它是会自适应的软惩罚，不是每次都严格满足的 hard constraint。

### 5.2 [Clipped surrogate objective](academic-term-lookup:clipped%20surrogate%20objective)

*首次完整讲解：Lecture 6 §5.2「Clipped surrogate objective」。*


Clipped objective 并不是直接让 dπold ≈ dπθ​​

它做的是：

限制 πθ 相对于 πold 的动作概率不要变化太猛， 只要策略每步不要变化太猛，总体状态就不会有太大变化
​
它通过r * A，来进行动作的校准，通过限制r，来做到状态分布的校准

令 $\pi_{\theta_k}$ 是收集当前 batch 的 old policy，$\pi_\theta$ 是正在优化的 current policy。对 batch 中每个 $(s_t,a_t)$：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_k}(a_t\mid s_t)}.
$$
[!example]

>	采数据时 old policy：$\pi_{\text{old}}(a_t|s_t)=0.5$
>	
>	现在 current policy：$\pi_\theta(a_t|s_t)=0.6$πθ​(at​∣st​)=0.6
>	
>	那么：$r_t = \frac{0.6}{0.5} = 1.2$
>	
>	表示这个动作的概率提高了：20%


不 clipping 时的 surrogate：

单个样本：$\boxed{r_t(\theta)\hat A_t}$​

如果A大于0， 那么这个动作是好动作， 也就希望提高这个动作发生的概率， 也就是使得rt>1。

但如果不进行clip， 则会疯狂的加大r，导致某个动作概率疯狂提高，也就使得policy update过大。

所以 PPO clipped objective 是

$$
\mathcal L_{\theta_k}^{\mathrm{CLIP}}(\theta)
=
\mathbb E_{\tau\sim\pi_{\theta_k}}
\left[
\sum_t
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

我们关心的始终是：J(πθ)−J(πold)​

但是这里 PPO **不直接优化这个真实值**。

它优化的是一个 surrogate：
$$\boxed{ L^{CLIP}(\theta) }$$​
它提高了，J(πθ)就能提高。


原始收益rA  截断后的收益 clip(rt​,0.8,1.2)A^t ，两者取min​，选择保守的那个。强行截断r

他作用是： 当 rt 朝“有利于提高 objective”的方向跑得过远时，不再提供额外奖励

`min` 选择较保守的收益：

- 若 $\hat A_t>0$，提高该动作概率是有利方向；当 $r_t>1+\epsilon$ 后，objective 进入平台，不再奖励继续增大概率。
- 若 $\hat A_t<0$，降低该动作概率是有利方向；当 $r_t<1-\epsilon$ 后，objective 进入平台，不再奖励继续减小概率。
- Clipping 只截断“继续朝有利方向走得过远”的激励；朝不利方向移动仍会被 objective 惩罚。

![[lec6-ppo-clipped-objective-p48.png|900]]

*图：左侧是 $\hat A>0$，ratio 超过 $1+\epsilon$ 后 objective 变平；右侧是 $\hat A<0$，ratio 低于 $1-\epsilon$ 后 objective 变平。来源：`lecture/lecture6post.pdf`，物理 PDF 第 48 页。*

| 情况                |       $\epsilon=0.2$ 的自拟数值 | Clipped objective 单样本值 | 含义                    |
| ----------------- | -------------------------: | ---------------------: | --------------------- |
| $\hat A=2,r=1.3$  |     $\min(2.6,1.2\times2)$ |                  $2.4$ | 好动作概率已增大过多，不再奖励超过 1.2 |
| $\hat A=-2,r=0.7$ | $\min(-1.4,0.8\times(-2))$ |                 $-1.6$ | 差动作概率已降低过多，不再奖励低于 0.8 |
| $\hat A=2,r=0.7$  |            $\min(1.4,1.6)$ |                  $1.4$ | 好动作概率下降是坏方向，不屏蔽惩罚     |
| $\hat A=-2,r=1.3$ |          $\min(-2.6,-2.4)$ |                 $-2.6$ | 差动作概率上升是坏方向，不屏蔽惩罚     |

最外层的E，不是把所有 trajectory 都列出来，一般会用采样平均，也就是Monte Carlo / minibatch estimate


> [!important] PPO 中的三个 policy 角色
> `old policy` $\pi_{\theta_k}$ 负责采集 batch，并作为 ratio 的冻结分母；`current policy` $\pi_\theta$ 是多个 gradient step 中不断变化的分子；$\hat A_t$ 也应视为在 old-policy batch 上预先估计并冻结的权重。每轮结束后再用新参数替换 old policy，而不是在每个 minibatch 内同步分母。


### 5.3 PPO 做到了什么，没有保证什么

Clipping 让同一个 near-on-policy batch 可以做多个 minibatch update，并降低单样本 ratio 继续向极端方向移动的激励。但它不等于：

- 对所有状态都施加严格 KL constraint；
- 保证每次真实 return 单调提高；
- 允许无限复用任意陈旧 replay data；
- 自动解决 advantage estimate 的 bias 或 critic 误差。

因此 PPO 常被称为 practical approximate trust-region method。它比一次性 REINFORCE 更能复用当前 batch，但仍依赖周期性地用新策略重新采样。

### 5.4 Assignment 2 中的 `PPO.update_policy`

Assignment 2 starter code 已缓存 rollout 时的 `old_logprobs`。实现对应关系是：

| 数学对象 | 代码对象 | 形状与操作 |
|---|---|---|
| $\log\pi_{\theta_k}(a_t\mid s_t)$ | `old_logprobs` | `[batch]`，rollout 时缓存，policy update 中作为常数 |
| $\log\pi_\theta(a_t\mid s_t)$ | current distribution `log_prob(actions)` | `[batch]`；连续动作须先对 action event 维聚合 |
| $r_t(\theta)$ | `torch.exp(new_logprobs - old_logprobs)` | `[batch]`；比直接除 probability 更稳定 |
| $r_t\hat A_t$ | unclipped surrogate | `[batch]` |
| $\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t$ | clipped surrogate | `[batch]` |
| $\min(\cdot,\cdot)$ | `torch.minimum` 或 `torch.min` | elementwise 后对 batch 求 mean |
| 最大化 $\mathcal L^{\mathrm{CLIP}}$ | optimizer loss | 对 surrogate mean 取负，再 `backward()` 和 `step()` |

推荐计算骨架：

~~~~python
new_logprobs = distribution.log_prob(actions)
ratios = torch.exp(new_logprobs - old_logprobs)
surrogate_1 = ratios * advantages
surrogate_2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
loss = -torch.minimum(surrogate_1, surrogate_2).mean()
~~~~

> [!warning] Starter code 的 advantage 注释有形状歧义
> `ppo.py:update_policy` docstring 把 `advantages` 写成 `[batch size, 1]`，但 `PolicyGradient.calculate_advantage` 与其他 update 接口使用 `[batch]`。实际实现应保持 `new_logprobs`、`old_logprobs`、`ratios`、`advantages` 都为 `[batch]`；若让 `[batch] * [batch,1]` 广播，会意外生成 `[batch,batch]`。

REINFORCE 不必缓存 old log-probability，因为每批只在同一当前 policy 下更新一次；PPO 要在参数已变化后继续计算新旧 ratio，所以必须保留 rollout policy 的 log-probability，或保留一份冻结 old policy 重新计算它。

## 6. Assignment Readiness：Assignment 2

### 6.1 现在已经覆盖的部分

- §2(a)：用反向递推 $G_t=r_t+\gamma G_{t+1}$ 在 $O(T)$ 计算 returns；首次完整讲解见 Lecture 5 §5.1。
- §2(b)：PPO clipped loss 的零梯度平台；见本讲 §5.2 的正负 advantage 分段解释。
- §2(c)：为什么 PPO 缓存 old log-probability，而 REINFORCE 不需要；见本讲 §5.4。
- §2 coding：`PPO.update_policy` 的 ratio、clip、minimum 与 loss 符号已经映射到 starter code。
- §3(a)--(c)：trajectory distribution 与 discounted state distribution；分别见 Lecture 5 §3.1 和本讲 §4.1。
- §3(d)：performance difference lemma 及 telescoping 证明；见本讲 §4.2。

现在可以开始 Assignment 2 的 policy-gradient、PPO 和 policy-induced distribution 部分。这表示前置知识已覆盖，不表示实现和证明已经熟练。

### 6.2 仍需单独阅读的部分

- Assignment 2 的 human-subjects ethics 问题需要阅读题面给出的 Belmont Report、IRB 与研究合规材料；Lecture 6 不覆盖这些规范性内容。
- 训练实验要求 21 次 run、三类环境与多 seed 汇总；运行成本、旧依赖兼容性和结果波动需要在实际作业阶段单独处理。
- Mastery evidence：目前没有记录独立推导、quiz、`PPO.update_policy` 测试或 benchmark 通过结果。

推荐下一步：先手算 §9 自测题 4、7、8，再实现 `PPO.update_policy`；用长度为 4 的 tensor 检查 ratio、正负 advantage clipping 和所有中间形状均为 `[4]`。

## 7. 本讲必会公式

1. 单个梯度项的方差最优 state baseline（首次推导：§2.2）：

   $$
   b_{\mathrm{term}}^*(s)
   =
   \frac{\mathbb E[\lVert\nabla\log\pi\rVert^2G_t\mid s]}
   {\mathbb E[\lVert\nabla\log\pi\rVert^2\mid s]}
   \approx V^\pi(s).
   $$

2. Advantage 与一步 actor-critic target（首次讲解：§2.4）：

   $$
   A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),
   \qquad
   \hat A_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t).
   $$

3. N-step target（来源补充：§2.5）：

   $$
   \hat Q_t^{(n)}
   =\sum_{k=0}^{n-1}\gamma^k r_{t+k}
   +\gamma^n\hat V(s_{t+n}).
   $$

4. Discounted state distribution（首次讲解：§4.1）：

   $$
   d^\pi(s)
   =(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr(s_t=s\mid\pi).
   $$

5. Performance difference lemma（首次推导：§4.2）：

   $$
   J(\pi')-J(\pi)
   =
   \frac{1}{1-\gamma}
   \mathbb E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)].
   $$

6. 单步 importance ratio（首次讲解：§4.3）：

   $$
   r(s,a)=\frac{\pi'(a\mid s)}{\pi(a\mid s)}.
   $$

7. Surrogate objective（首次讲解：§4.4）：

   $$
   \mathcal L_\pi(\pi')
   =
   \frac{1}{1-\gamma}
   \mathbb E_{s\sim d^\pi,a\sim\pi}
   \left[r(s,a)A^\pi(s,a)\right].
   $$

8. Policy KL（首次讲解：§4.5）：

   $$
   D_{\mathrm{KL}}(\pi'\Vert\pi)[s]
   =
   \sum_a\pi'(a\mid s)\log\frac{\pi'(a\mid s)}{\pi(a\mid s)}.
   $$

9. PPO clipped objective（首次讲解：§5.2）：

   $$
   \mathcal L^{\mathrm{CLIP}}(\theta)
   =
   \mathbb E
   \left[
   \min\left(r_t(\theta)\hat A_t,
   \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
   \right].
   $$

## 8. 容易混淆点

1. **任意 baseline 无偏 vs 最优 baseline**：action-independent 保证期望不变；只有与 return 合理相关的 baseline 才能有效降方差。
2. **$V^\pi(s)$ vs 单项最优 baseline**：后者是 score-norm 加权条件均值；$V^\pi(s)$ 是常用近似。跨时间协方差存在时，不能无条件称其为整个 trajectory estimator 的全局最优解。
3. **Return residual vs true advantage**：$G_t-b_\phi(s_t)$ 是 sampled estimate；$A^\pi=Q^\pi-V^\pi$ 是条件期望之差。
4. **参数步长 vs policy-space 距离**：$\lVert\Delta\theta\rVert$ 小不保证所有状态下的动作分布变化小。
5. **$d^\pi$ vs 单步 state distribution**：$d^\pi$ 是跨所有时间步的 normalized discounted mixture。
6. **旧策略 vs 新策略 advantage**：performance difference lemma 用 $A^\pi$ 评价候选 $\pi'$，不能随意把 advantage 上标换成 $\pi'$。
7. **动作 ratio vs 完整 distribution correction**：$\pi'/\pi$ 只校正动作；把 $d^{\pi'}$ 换成 $d^\pi$ 仍是近似。
8. **KL 的方向**：$D_{\mathrm{KL}}(\pi'\Vert\pi)$ 与 $D_{\mathrm{KL}}(\pi\Vert\pi')$ 一般不同，必须按公式核对谁在期望外层。
9. **PPO clipping vs hard constraint**：clipping 限制 objective 的激励，不保证实际 ratio 或平均 KL 永远留在阈值内。
10. **PPO vs arbitrary off-policy replay**：PPO 有限复用最近 old policy 的 batch；它不是把任意历史数据无限放入 replay buffer。
11. **Old log-probability vs current log-probability**：分母必须冻结为 rollout 时的值；若用当前网络重算两者，ratio 会错误地恒为 1。
12. **Advantage shape**：policy log-probability 与 advantage 都应为 `[batch]`；`[batch,1]` 会触发危险广播。
13. **Pre vs post deck**：Lecture 6 课堂 post deck 到物理第 48 页结束；§2.5 明确来自 `lecture/lecture5pre.pdf` 第 72--75 页，仅作来源补充。

## 9. 自测题

### 题目

1. 为什么 $V^\pi(s)$ 只是常用 baseline？§2.2 的加权解在哪个作用域内是精确最优？
2. Baseline 不依赖动作为什么只保证无偏，却不保证降方差？
3. 为什么 Adam 调 learning rate 仍没有直接解决 policy-space 步长问题？
4. 从 $A^\pi(s_t,a_t)$ 的 Bellman 形式出发，说明 performance difference lemma 中哪些 value 项会 telescoping。
5. 单步 importance ratio 已经修正了什么 mismatch？还留下什么 mismatch？
6. 为什么 surrogate objective 只在新旧策略接近时更可信？
7. 令 $\epsilon=0.2$、$\hat A=3$、$r=1.4$，PPO 单样本 clipped objective 是多少？梯度为什么在继续增大 $r$ 的方向上为 0？
8. 令 $\epsilon=0.2$、$\hat A=-3$、$r=0.6$，PPO 单样本 clipped objective 是多少？
9. 为什么 `exp(new_logprob - old_logprob)` 比直接先 exponentiate 再相除更合适？
10. PPO 是否保证每次 update 后真实 return 单调增加？为什么？

<details>
<summary>查看答案</summary>

1. 固定状态、只优化单个 score-return 梯度项时，精确解按 $\lVert\nabla\log\pi\rVert^2$ 加权 return；score norm 变化不大时才近似 $V^\pi(s)$。多个时间步之和还包含交叉协方差，因此该结论不能无条件推广为整个 trajectory estimator 的全局最优解。
2. 无偏性只用到 baseline 项对动作期望为 0；方差取决于它是否能让 $G_t-b(s_t)$ 的梯度加权样本更集中。
3. Adam 控制的是参数坐标中的 update；神经网络参数到动作分布的映射是非线性的，同样参数步长可对应不同 policy change。
4. $\gamma^t\gamma V^\pi(s_{t+1})$ 与下一项的 $-\gamma^{t+1}V^\pi(s_{t+1})$ 相消，只留下 discounted rewards、初始 $-V^\pi(s_0)$ 与消失的尾项。
5. 它修正固定状态下 $a\sim\pi$ 到 $a\sim\pi'$ 的动作分布差异；状态仍来自 $d^{\pi'}$，替换成 $d^\pi$ 是额外近似。
6. 新旧策略接近时 discounted state distributions 更接近，KL performance bound 也让 surrogate approximation error 更小。
7. $\min(1.4\times3,1.2\times3)=3.6$。平台项被选中，继续增大 ratio 不再改变目标。
8. $\min(0.6\times(-3),0.8\times(-3))=\min(-1.8,-2.4)=-2.4$。
9. Log-probability 差直接给 log ratio，避免分别得到很小 probability 后再相除造成 underflow 或数值不稳定。
10. 不保证。Clipping 是 samplewise surrogate 的软限制，不是对真实性能或全状态 KL 的 hard monotonic-improvement constraint；critic error、采样噪声和优化误差仍存在。

</details>

## 10. 本讲小结

Lecture 6 先把 baseline 从“无偏技巧”推进到“方差优化问题”：固定状态下单个梯度项的最优 baseline 是 score-norm 加权的 return 条件均值，$V^\pi(s)$ 是实用近似。VPG 把 policy update 与 baseline regression 接成完整循环；critic 再用 value estimate 或 bootstrap target 提供 advantage estimate，形成 actor-critic。

Vanilla policy gradient 的真正瓶颈不只是一阶梯度 noisy。它依赖当前策略数据，并且参数空间步长无法可靠代表 policy-space 变化。Performance difference lemma 用旧策略 advantage 精确表达新旧性能差；单步 ratio 与 $d^{\pi'}\approx d^\pi$ 把它变成可由旧数据估计的 surrogate，而 KL bound 说明新旧策略接近时这个近似更可信。

PPO 最终把上述思想变成工程上简单的 clipped objective：有利方向的 ratio 变化超过阈值后不再得到额外激励，因此最近一批 old-policy 数据可以被有限复用。它提高实用稳定性和样本利用率，但不是 hard trust-region guarantee，也不是任意 off-policy replay 方法。

## 11. 延伸阅读

### 经典基础

- Kakade and Langford (2002), *Approximately Optimal Approximate Reinforcement Learning*：conservative policy improvement 与课件 performance bound 路线的基础。
- Schulman et al. (2015), *Trust Region Policy Optimization*：以 policy-space trust region 约束更新的经典工作。
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*：adaptive KL penalty 与 clipped surrogate objective 的原始论文。
- Achiam et al. (2017), *Constrained Policy Optimization*：课件引用的 relative policy performance bound 与受约束策略更新。

### 前沿动态

截至 2026-07-28 核实：本讲不额外加入前沿项目。当前目标是掌握 performance difference、surrogate、KL 与 clipping 之间的推理链；RLHF 和现代大模型中的 PPO 变体留到课程明确进入相应主题时再结合一手资料展开。
