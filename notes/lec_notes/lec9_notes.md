---
title: CS234 Lecture 9 - Data-Efficient Reinforcement Learning
aliases:
  - CS234 Lec9
tags:
  - cs234
  - reinforcement-learning
  - multi-armed-bandit
  - regret
  - ucb
---

# CS234 Lecture 9 Notes: Data-Efficient Reinforcement Learning

来源：lecture/lecture9post.pdf，CS234 Winter 2026，Emma Brunskill；共 53 个物理 PDF 页面，部分页面沿用课件内部的动画页码。

笔记规范：cs234-rl-tutor v2。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-13（本讲延伸阅读只列课件直接引用的经典来源）。

## 0. 本讲覆盖清单

课件前半包含复习题、目录页和重复动画；下面按物理 PDF 页面记录覆盖位置。

- [x] 第 1--3 页：标题、RLHF/DPO 复习题及答案；写入 §1 和 §9。
- [x] 第 4--8 页：课程位置、RL 难点、评价标准、setting/framework/approach 和本讲路线；写入 §1。
- [x] 第 9--12 页：multi-armed bandit 定义、broken-toe 说明例子及理解检查；写入 §2。
- [x] 第 13--17 页：greedy 的 Monte Carlo 估计、一次错误观测导致的锁定；写入 §3。
- [x] 第 18--25 页：regret 定义、gap/count 分解、greedy 的线性 regret 和不可直接观测性；写入 §4。
- [x] 第 26--32 页：$\varepsilon$-greedy、示例、线性 regret 检查题；写入 §5。
- [x] 第 33--35 页：sublinear regret、problem-independent/dependent bounds、Lai--Robbins 下界；写入 §6。
- [x] 第 36--39 页：optimism under uncertainty 和 UCB 抽象；写入 §7。
- [x] 第 40--43 页：sub-Gaussian concentration、union bound 补充、UCB1；写入 §7--§8。
- [x] 第 44--50 页：UCB toy walkthrough、regret 表、confidence-level 说明和 Lecture 10 proof 预告；写入 §8。
- [x] 第 51--53 页：lower-confidence-bound 思考题、课程收束和下讲预告；写入 §9--§10。

课件目录页列出 Bayesian regret 和 probability matching / Thompson sampling，但本 post deck 没有展开相应定义或算法；它们只作为后续方向记录在 §1，不把未讲内容伪装成本讲结论。

## 1. 本讲主线

与 Lecture 8 的关系：Lecture 8 研究的是从人类比较反馈学习一个可优化的代理目标；本讲转向更基础的探索问题：当每个动作的奖励分布未知时，怎样用尽量少的试验找到高回报动作。这里先用单步决策的 bandit 隔离探索，下一讲再继续讨论更快的学习与 Bayesian 视角。

前面更多在问：

> **reward 到底是什么？怎么从人类偏好、专家行为里把 reward 学出来？**

这一章开始问另一个问题：

> **假设每个动作确实都有某个真实的平均 reward，但我不知道它是多少。我要怎么一边尝试、一边尽快找到最好的动作？**

这就是 **exploration vs. exploitation（探索 vs. 利用）**

**本讲路线图**

1. 把未知的动作回报形式化成多臂 bandit，并先看纯 greedy 为什么会被偶然样本误导。
2. 用 regret 量化探索期间相对于最优动作的机会损失。
3. 比较固定 $\varepsilon$ 的随机探索与 sublinear regret 的目标，说明探索和利用不能简单二选一。
4. 用 optimism under uncertainty 把不确定但可能很好的动作纳入选择，得到 UCB 的基本形式。
5. 用集中不等式和 union bound 解释 UCB1 的置信半径；完整 regret proof 在 Lecture 10 修正并展开。

本讲的核心不是某个算法永远选最优臂，而是：在未知环境中，算法应尽快减少对高损失动作的重复试验，同时保留足够的信息收集。 也就是在尽量少的次数中，探索出最优reward

## 2. 多臂 bandit：把探索问题隔离出来

多臂 bandit 是比 MDP 更简单的决策 setting：**每一轮只选一个动作并得到一个奖励**，不需要处理状态转移或延迟的后续状态。它适合描述每位病人只接受一次治疗，然后观察结果这样的单次选择问题。

### 2.1 形式化定义

多臂 bandit 可以写成二元组 $(\mathcal A,\mathcal R)$：

- $\mathcal A$ 是已知的 $m$ 个动作（arms）的集合；
- 对每个动作 $a$，$\mathcal R_a$ 是未知的奖励分布，一个action不可能永远对应一个固定的reward；
- 第 $t$ 轮选择 $a_t\in\mathcal A$，环境从 $\mathcal R_{a_t}$ 采样 $r_t$；
- 目标是在 $T$ 轮内最大化累计奖励。

$$
a_t\in\mathcal A,\qquad
r_t\sim\mathcal R_{a_t},\qquad
\max \mathbb E\left[\sum_{t=1}^{T}r_t\right].
$$

这里每个动作的真实均值是

$$
Q(a)=\mathbb E[r\mid a].
$$
>	如果无限次执行 action a，平均 reward 是多少？

$Q(a)$ 是一个标量动作价值，不是 Lecture 1 中依赖状态的 $Q^\pi(s,a)$；bandit 可以看成一个只有一个状态、但有多个动作的特殊 MDP。关键困难是 $Q(a)$ 未知，算法必须边试边估计。

### 2.2 说明例子：broken-toe treatment

课件构造了一个教学例子：治疗骨折脚趾有三个动作，$a_1$ 为手术，$a_2$ 为 buddy taping，$a_3$ 为不处理；六周后用 X-ray 判断是否愈合，奖励为愈合 $1$、未愈合 $0$。每个动作的奖励可以建模为参数未知的 [Bernoulli](academic-term-lookup:bernoulli) 分布：

$$
r_t\mid a_i\sim\operatorname{Bernoulli}(\theta_i),\qquad
Q(a_i)=\theta_i.
$$
在第 t 步选择了第 i 个臂 动作 ai​ 的条件下，获得的奖励 rt​ 服从参数为 θi​ 的伯努利分布

$\theta_i$：就是使用第 i 种治疗后病人痊愈的真实概率。

这是课件明确标注的说明性虚构案例，$\theta_i$ 不是实际医疗疗效。这里适合用 bandit，是因为一次治疗对应一次动作；如果治疗过程中还要连续决策，才更像 MDP。

如果 $0<\theta_i<1$，重复选择同一个动作时有时得到 $1$、有时得到 $0$。一次观测是随机样本，不等于动作的真实均值。

$\theta_i$ 是未知的，你只能靠真实病人不断试验

## 3. Greedy：只相信当前估计会发生什么

与 Lecture 4 的 $\varepsilon$-greedy 控制关系：Lecture 4 已经使用过 $\varepsilon$-greedy；这里先把其中的 greedy 部分单独抽出来，观察完全不探索的策略为何会失败。

### 3.1 Monte Carlo 动作价值估计

这里的 Monte Carlo：

> **通过实际采样得到的 reward，取平均值来估计真实期望。**


设 $N_t(a)$ 表示截至选择第 $t$ 个动作前，动作 $a$ 被选择的次数。用所有观测到的奖励平均估计动作均值：
$$
\widehat Q_t(a)
=
\frac{1}{N_t(a)}
\sum_{i=1}^{t}r_i\mathbf 1(a_i=a),
\qquad N_t(a)>0.
$$

估计量的输入是动作 $a$ 的历史奖励样本，输出是一个标量均值估计。纯 greedy 在第 $t$ 轮选择

$$
a_t=\arg\max_{a\in\mathcal A}\widehat Q_{t-1}(a).
$$

若尚未拉过某个 arm，实际实现需要先初始化每个 arm，或约定未访问动作的估计为 $+\infty$；课件的 toy example 采用先把每个 arm 采样一次的初始化。

> [!example] 具体计算：一次坏运气如何锁定 greedy
> 课件设真实均值为 $Q(a_1)=0.95$、$Q(a_2)=0.90$、$Q(a_3)=0.10$，并先各采样一次。若观测结果是
>
> $$
> r_1=0,\qquad r_2=1,\qquad r_3=0,
> $$
>
> 则当前估计为 $\widehat Q(a_1)=0$、$\widehat Q(a_2)=1$、$\widehat Q(a_3)=0$。greedy 会一直选择 $a_2$；如果之后每次 $a_2$ 都得到 $1$，$a_1$ 就再也没有新数据来纠正第一次的 $0$。
>
> 这不是说 $a_2$ 一定会永久最差，而是说明从未重新观察的动作无法靠自身的历史估计恢复。一次偶然的初始样本就可能把算法锁在次优动作上。


### 3.2 Greedy 的边界

greedy 的优点是实现简单、利用当前估计直接选择最高均值；
但它没有主动的信息获取机制。即使一个未充分尝试的 arm 真实均值很高，只要它早期样本偏低，greedy 也可能永远不再选择它。意思是本来选这个动作是最优的，但前几次采样你运气不好，这个动作的价值很低，会让greedy错过这个动作

后面更谨慎的方法都在处理同一缺陷：如何让不确定性影响动作选择。 


## 4. Regret：用机会损失评价学习过程

与前面只问是否收敛的标准不同，bandit 还要问：算法在学会之前浪费了多少本可以获得的奖励？regret 衡量的正是这段学习过程的机会成本。

在我“学习哪个 action 最好”的过程中，我因为选错 action 一共损失了多少 reward？为了学好，你付出了多大的代价

### 4.1 单步 regret 与总 regret

多臂 bandit每一轮只选一个动作并得到一个奖励

最优动作和最优均值分别是

$$
a^*\in\arg\max_{a\in\mathcal A}Q(a),
\qquad
V^*=Q(a^*)=\max_{a\in\mathcal A}Q(a).
$$

第 $t$ 轮的单步 regret 是相对于最优均值的期望机会损失：(每一轮都对应着同样的那几个动作)

$$
\ell_t=\mathbb E\left[V^*-Q(a_t)\right].
$$
第 t 轮，你实际选了：$a_t$

它的真实期望 reward 是：$Q(a_t)$

如果你知道真相，本来应该选 a∗，平均可以得到：$V^*$

于是这一轮因为没选最优 action 而损失：$V^*-Q(a_t)$

这就是这一轮的**机会损失**。


总 regret 为

$$
L_T
=
\mathbb E\left[\sum_{t=1}^{T}\left(V^*-Q(a_t)\right)\right].
$$

它不是把实际观测奖励 $r_t$ 与最优动作的实际奖励逐次相减，而是比较 期望动作价值。环境的随机奖励被期望 吸收，动作选择策略的随机性也包含在外层期望中，被外层期望考虑到

Regret **不是** 
$r_t^*-r_t$
这种实际采样 reward 之间的差。

而是在比较：
$V^*-Q(a_t)$
也就是**两个 action 的期望 reward 差距**。



最大化累计期望奖励与最小化总 regret 等价，因为 $TV^*$ 是固定基准(固定常数)：

$$
L_T
=TV^*-
\mathbb E\left[\sum_{t=1}^{T}Q(a_t)\right].
$$
假设你从第一轮开始就知道最优 action。

那么 T 轮可以获得的 expected reward 是：$TV^*$

而你的算法实际 expected cumulative reward 是：

$\mathbb E\left[\sum_{t=1}^{T}Q(a_t)\right]$

所以 regret：

$L_T=TV^*-\mathbb E\left[\sum_{t=1}^{T}Q(a_t)\right]$


### 4.2 Gap--count 分解

对每个动作定义 gap

> **这个 action 和最优 action 到底差多少**

$$
\Delta_a=V^*-Q(a)\ge 0,
$$

并令 $N_T(a)$ 为 $T$ 轮 中选择 $a$ 的次数。交换求和与按动作计数可以得到

$$
L_T
=
\sum_{a\in\mathcal A}\mathbb E[N_T(a)]\Delta_a.
$$
> 总 regret = 每个 action「选了多少次」×「每选错一次损失多少」，然后全部加起来。


这个分解揭示了算法的真正目标：大 gap 的动作每多试一次都很贵，因此应尽量少选；小 gap 的动作即使被多比较几次，损失也较小。但 $\Delta_a$ 本身未知，算法只能通过奖励样本间接推断。

> [!example] 具体计算：课件中的 greedy regret 表
> 仍使用 $Q(a_1)=0.95$、$Q(a_2)=0.90$、$Q(a_3)=0.10$，所以 $a_1$ 最优，三个 gap 为 $0$、$0.05$、$0.85$。若动作序列为 $a_1,a_2,a_3,a_2,a_2$，每一轮的期望 regret 是
>
> $$
> 0,\quad 0.05,\quad 0.85,\quad 0.05,\quad 0.05,
> $$
>
> 总和为 $1.00$。其中 $a_3$ 只选了一次，却贡献了大部分损失；这说明 gap 和 count 必须一起看。

### 4.3 Regret 的可观测性与线性 regret

真实 bandit 中的 $V^*$ 是未知的，所以实践中不能直接计算真实 regret；理论分析通常给出算法在任意满足假设的 bandit 上的 **regret 上界**。

如果一个次优动作被以固定正比例选择，则 $\mathbb E[N_T(a)]=\Theta(T)$，且 $\Delta_a>0$，于是总 regret 是线性的：$L_T=\Theta(T)$。

Θ(T)：
随着 T 增大，regret 和 T 同一个数量级增长。


纯 greedy 可能因为初始样本锁定次优动作，因此可能有线性 regret。

但我们真正想要：
LT随着时间的增加，它增长得**越来越慢**

也就是：
$L_T=o(T)$

这叫 **sublinear regret**。


## 5. $\varepsilon$-greedy：固定比例的随机探索

### 5.1 算法与选择概率

$\varepsilon$-greedy 以概率 $1-\varepsilon$ 选择当前估计最高的动作，以概率 $\varepsilon$ 从所有动作中均匀随机选择：

$$
a_t=
\begin{cases}
\arg\max_{a\in\mathcal A}\widehat Q_{t-1}(a),&\text{with probability }1-\varepsilon,\\
\operatorname{Uniform}(\mathcal A),&\text{with probability }\varepsilon.
\end{cases}
$$

当 $\varepsilon$ 固定且存在次优动作时，算法会持续探索，因此从长期平均看会以正比例选择次优动作。这提供了纠正早期估计错误的机会，但也意味着长期 regret 通常仍是线性的。

> [!example] 具体计算：$\varepsilon$-greedy 的下一步概率
> 课件示例中各 arm 已采样一次，观测为 $1,1,0$，因此 $a_1$ 和 $a_2$ 并列最大；设 $\varepsilon=0.1$，并列时均匀拆分。于是
>
> $$
> P(a_1)=0.9\times\frac12+0.1\times\frac13=0.4833\ldots,
> $$
> $$
> P(a_2)=0.9\times\frac12+0.1\times\frac13=0.4833\ldots,
> $$
> $$
> P(a_3)=0.1\times\frac13=0.0333\ldots.
> $$
>
> 所以 $a_3$ 仍有机会被重新观察，但固定 $\varepsilon$ 也会让明显次优的动作持续产生损失。


### 5.2 线性 regret 检查

课件的理解检查假设存在某个 $\Delta_a>0$ 的次优动作：

- $\varepsilon=0.1$：随机探索以正比例抽到它，可能产生 线性 regret；
- $\varepsilon=0$：退化为纯 greedy，也可能因锁定次优动作而产生 线性 regret。

“永远探索”和“永不探索”都可能线性 regret，说明目标不是在二者之间二选一，而是让探索次数随信息增加而变得有针对性。

为什么固定 ϵ 会导致 linear regret？

假设有一个次优 action a，它的 gap：$\Delta_a>0$

在 exploration 阶段，每次都有固定概率抽到它。

假设一共有 m 个 action。

那么每一轮由于 exploration 选到 a 的概率至少大约是：$\frac{\epsilon}{m}$

所以运行 T 轮以后，大约会选它：

$\mathbb E[N_T(a)]\approx\frac{\epsilon}{m}T$

也就是：
$\mathbb E[N_T(a)]=\Theta(T)$

而之前我们刚学过：​

$L_T=\sum_{a\in\mathcal A}\mathbb E[N_T(a)]\Delta_a$

对于这个次优 action：

$\mathbb E[N_T(a)]\Delta_a=\Theta(T)\Delta_a$

因为 Δa​ 是固定正数，所以：

$L_T=\Theta(T)$

这就是 **linear regret**。



## 6. Sublinear regret 与难度下界

真正理想的目标：Sublinear Regret

我们希望：$L_T$

可以增长，因为探索不可避免。

但希望它增长得比 T 慢。

也就是：$L_T=o(T)$

等价地：$\frac{L_T}{T}\rightarrow0$

这就是 **sublinear regret**

$\frac{L_T}{T}$ 可以理解成：

> **平均每轮损失多少 reward。**

如果：
$\frac{L_T}{T}\rightarrow0$

说明随着算法学习越来越久：

> 平均每一步造成的机会损失越来越小。

也就是算法越来越接近：

几乎一直选择最优 action



### 6.1 两类 regret 上界

这一节开始进入：

> **我们怎么从理论上证明一个 bandit 算法到底有多好？**

理想情况下，总 regret 随时间增长慢于线性，即

$$
\frac{L_T}{T}\longrightarrow 0.
$$
有两类上界：

1. **Problem-independent bound**：不考虑具体每个 action 到底差多少，只用时间步数 $T$、动作数等规模参数描述最坏情况的增长。（不管你的 bandit 具体长什么样，这个 bound 都成立）
2. **Problem-dependent bound**：利用具体 bandit 的 gap 或分布相似性，说明每个动作被选了多少次以及这些选择带来的损失。

前者便于比较不同问题上的统一最坏情况，后者能体现 最优臂和次优臂 很接近时 更难区分的实例难度。

如果：
$Q(a_1)=0.90,\qquad Q(a_2)=0.89$

那么：

$\Delta_{a_2}=0.01$

两者非常接近。

这时因为 reward 有随机噪声：

> 你必须采很多很多次，才能有把握说 a1​ 真的比 a2​ 好。

所以：

$\Delta_a\text{ 越小}$

往往代表：

> **这个 action 越难和最优 action 区分。**




### 6.2 Lai--Robbins 下界

课件用 Lai--Robbins 理论说明：对随机 bandit，任何足够好的算法都不能把所有 次优动作 只试一个常数次；在适当正则条件下，次优动作的探索次数至少按对数增长。


对于每一个次优 action a，你至少还得尝试它大约：

$O(\log T)$次

所以最好的 regret 通常也不可能做到：$O(1)$

因为次优和最优差距很小的话，无法通过太少的次数判断出要选哪个，因为担心所抽取的样本很可能是随机噪声



Lai–Robbins 最经典的结论是：对于次优动作 a，探索次数至少要按对数级增长，大致是
​
$\mathbb E[N_T(a)]\gtrsim\frac{\log T}{D_{\mathrm{KL}}(\mathcal R_a\Vert\mathcal R_{a^*})}$

- NT​(a)：前 T 轮里 action a 被选了多少次
- a∗：真正的最优 action
- Ra​：action a 的 reward distribution
- DKL​：两个 reward distribution 有多容易区分

更严谨的写：
$$\liminf_{T\to\infty}\frac{\mathbb E[N_T(a)]}{\log T}\ge\frac{1}{D_{\mathrm{KL}}(\mathcal R_a\Vert\mathcal R'_a)}$$
要证明一个 arm 不是最优的，你至少要从它那里收集 logT 量级的统计证据。


再由此从采样次数下界 推出 regret 的下界：

$$L_T=\sum_{a:\Delta_a>0}\Delta_a\mathbb E[N_T(a)]$$

除以：$\log T$

得到：

$$\frac{L_T}{\log T}=\sum_{a:\Delta_a>0}\Delta_a\frac{\mathbb E[N_T(a)]}{\log T}$$

再把刚才证明的代进去：

$$
\liminf_{T\to\infty}
\frac{L_T}{\log T}
\ge
\sum_{a:\Delta_a>0}
\frac{\Delta_a}{D_{\mathrm{KL}}(R_a\,\|\,R_{a^*})}.
$$

也就是说：

> **任何足够好的算法，其 regret 至少要以 logT 的规模增长。**

所以：$O(\log T)$

不是因为算法“不够好”。

恰恰相反：

> 在很多 stochastic bandit setting 下，logT 已经接近理论上最好的量级。



$\liminf_{T\to\infty}\frac{L_T}{\log T}$

你可以暂时把它理解成：

> 当 T 非常非常大的时候，看 LT​/logT 最终至少会维持在什么水平。

它是在描述一个**渐近下界**。长期来看，你无法把 regret 的增长压到这个尺度以下

求和只对 Δa​>0：

$\Delta_a=V^*-Q(a)$
$\Delta_a>0$的 次优action 才是要讨论的，只有它才能产生regret（除了最优动作，其他都是次优）



课件页面使用的是结构性表达，精确适用条件依赖具体 bandit 假设。分母的 KL 散度衡量 次优臂分布与最优臂分布的可区分程度；分布越相似，越需要更多样本才能放心停止探索。这个下界仍是 sublinear 的，因此学得越来越好并不意味着完全没有探索损失。



## 7. Optimism under uncertainty：用不确定性促成信息获取

ϵ-greedy，无论我们知道多少信息，永远都有固定概率乱探索

那能不能不“随机乱探索”，而是**专门探索那些我们还不确定、但有可能很好**的 action？

>	面对不确定性采取乐观估计

### 7.1 乐观原则

optimism in the face of uncertainty 的动作选择原则是：优先选择那些在某个合理的未知真值下 可能 很有价值的动作。它有两种结果：

- 如果该动作确实高回报，尝试本身就带来较高奖励；
- 如果它实际回报较低，这次尝试会降低其估计均值或不确定性，帮助算法排除它。

因此探索不再是与当前估计无关的均匀随机，而是由可能的最好表现驱动。

假设有两个 action：

$\hat Q(a_1)=0.8$

而且已经试了 1000 次，所以非常确定。

另一个：

$\hat Q(a_2)=0.7$

但是只试过 2 次。

Pure greedy 会直接说：

$0.8>0.7$

所以选 a1​。

但 UCB 会想：

> a2​ 虽然目前均值只有 0.7，但我只试了两次，它的真实 Q(a2​) 会不会其实是 0.9，只是现在还没看出来？

因此不给 action 只看：$\hat Q(a)$

而是看：

当前估计+不确定性 bonus， 如果不确定性大，代表我们取得样本就很少，所以就算它当前估计值不高，我们也可以选这个动作，因为它真实值有较大几率会变高

所以选择动作，我们就比较当前估计+不确定性 bonus哪个大
​

### 7.2 Upper confidence bound

我们一般就用置信上界

对每个动作构造 UCB 上置信界 $U_t(a)$，希望在高概率下真实均值满足

$$
Q(a)\le U_t(a).
$$

然后选择上置信界最大的动作：

$$
a_t=\arg\max_{a\in\mathcal A}U_t(a).
$$

$U_t(a)$ 通常由 经验均值 加一个随样本数下降的 bonus 组成：样本少的动作 bonus 大，样本多的动作 bonus 小。这使同一个选择规则同时表达 exploitation 和 exploration。

$U_t(a)$ 是：

> **根据目前数据，Q(a) 合理情况下“最多可能有多好”。**

是估计值再加一个 上置信 bonus。

$$U_t(a)=\hat Q_t(a)+\text{bonus}_t(a)$$

第一部分： $\hat Q_t(a)$

大 → 说明已经观察到它 reward 很高。

这是：exploitation。

第二部分：$\text{bonus}_t(a)$

大 → 说明我们还很不确定。

这是：exploration。

所以 UCB 非常漂亮的一点就是：

同一个分数=利用价值+探索价值​

不需要再像 ϵ-greedy 一样单独扔硬币决定“这轮探索还是利用”。



比如：$\hat Q(a)=0.6$

同时算出一个置信半径：η=0.1

那么置信上界就是：

$\mathrm{UCB}(a)=\hat Q(a)+\eta=0.7$

它表达的是：

> 根据目前的数据，我们有较高置信度认为真实的 Q(a) 不会超过 0.7。

这个结论不是百分之百成立，而是例如以 95% 的置信度成立。

所以“置信上界”最直观的理解就是：

对真实值的一个“比较有把握的最高估计”

随着样本数的增多，置信半径会越来越小
​
## 8. 从集中不等式到 UCB1

这个bonus是怎么来的？

是从概率界推出来的

### 8.1 Sub-Gaussian 集中界

若独立随机变量 $X_i$ 的均值为 $\mu$，且噪声是 $\sigma$-sub-Gaussian，样本均值为

$$
\widehat\mu_n=\frac1n\sum_{i=1}^{n}X_i,
$$

则对任意 $\eta>0$，课件引用的 concentration inequality （集中不等式）给出

$$
P(\widehat\mu_n-\mu\ge\eta)
\le
\exp\left(-\frac{n\eta^2}{2\sigma^2}\right),
\qquad
P(\mu-\widehat\mu_n\ge\eta)
\le
\exp\left(-\frac{n\eta^2}{2\sigma^2}\right).
$$

$\mu\ge\hat\mu_n+\eta$

也就是说：

> 真实均值比我们的估计高出至少 η

η 就是：

> 我允许样本均值和真实均值差多少。

$n\eta^2$

随着 n 增大，右边指数下降得非常快。

所以：

> **样本越多，估计越可靠。**

---

从集中不等式推出 confidence bound：

令失败概率为 $\delta\in(0,1]$，把右侧设为 $\delta$

令：

$$\exp\left(-\frac{n\eta^2}{2\sigma^2}\right)=\delta$$
$\delta$ 的概率可能会超出置信区间


现在解 η。

两边取 log：

$$-\frac{n\eta^2}{2\sigma^2}=\log\delta$$

因为：

$$-\log\delta=\log\frac1\delta$$

所以：

$$\frac{n\eta^2}{2\sigma^2}=\log\frac1\delta$$

得到：
​$$\eta^2=\frac{2\sigma^2\log(1/\delta)}{n}$$

所以：​
$$\boxed{\eta=\sigma\sqrt{\frac{2\log(1/\delta)}{n}}}$$
这就是 UCB bonus 的来源。


可得到单侧上置信界：

$$\mu\le\hat\mu_n+\eta$$

把刚才的 η 代进去：
$$
\mu
\le
\widehat\mu_n
+\sigma\sqrt{\frac{2\log(1/\delta)}{n}}
\quad\text{with probability at least }1-\delta.
$$

如果需要同时控制上下偏差，使用 union bound，把每一侧的失败概率设为 $\delta/2$（我们希望总失败概率最多$\delta$，把它分给两个坏事件，偏高失败，偏低也失败）：

$$
|\widehat\mu_n-\mu|
\le
\sigma\sqrt{\frac{2\log(2/\delta)}{n}}
\quad\text{with probability at least }1-\delta.
$$

课件第 42 页的 post-class 注释正是在补充这一点；这里的 $\delta$ 是置信失败概率，不要和 $\varepsilon$-greedy 的探索率混淆。


### 8.2 UCB1 形式

是把上一页普通统计里的 confidence bound，正式改写成 **Bandit 第 t 轮真正能执行的 UCB 算法**

在奖励是 $1$-sub-Gaussian（$\sigma=1$，所以下面的公式里没有$\sigma=1$） 的约定下，课件给出 UCB1 的简化形式：

$$
a_t
=
\arg\max_{a\in\mathcal A}
\left[
\widehat Q_{t-1}(a)
+\sqrt{\frac{2\log(1/\delta)}{N_{t-1}(a)}}
\right].
$$
>	到了第 t 轮，我根据前 t−1 轮收集到的数据，为每个 action 算一个 UCB，然后选择 UCB 最大的 action，不再像之前一样，只关注Q了

$\text{UCB}=\text{当前均值估计}+\text{当前不确定性}$

有些版本把 $\delta$ 设置成随时间变化的失败概率，或把 $t$ 放进对数项；这些常数和时间调度会影响具体理论界，但不改变 UCB 的结构：经验均值加上 随访问次数衰减 的乐观 bonus。

> [!example] 过程追踪：UCB 如何重新检查不确定动作
> 课件的 UCB toy walkthrough 仍使用真实均值 $0.95,0.90,0.10$，先得到样本 $1,1,0$，使经验均值为 $1,1,0$。因为三个 arm 都只被访问一次，前两者的高均值和相同 bonus 形成并列；算法会选择其中一个，再用新奖励更新该 arm 的均值和访问次数。
>
> 无论第一个选择是 $a_1$ 还是 $a_2$，$a_3$ 的 bonus 仍会保持较大，直到它获得足够数据。与 greedy 不同，UCB 不会因为一次 $0$ 就永久放弃 $a_3$；它让“尚未知道”本身成为选择理由。

### 8.3 UCB regret 的直观结构

设次优臂 gap 为 $\Delta_a>0$。当一个臂已被采样很多次，它的 bonus 变小；若其真实均值确实低于最优臂，最终其 UCB 通常也会低于最优臂。于是大 gap 的臂只需较少样本就能被排除，而小 gap 的臂可能需要更多样本，这与 Lai--Robbins 下界中的可区分性决定难度相呼应。

Lecture 9 原本尝试给出较短的 UCB proof，但课后发现存在错误；Lecture 10 会改用 textbook Theorem 7.1 的 proof sketch。这里不把未完成的推导写成已证明的定理。

## 9. 复习与检查

### 9.1 复习题答案

Lecture 9 开头的 RLHF/DPO 复习题中，三条具体陈述均不成立：DPO 不训练一个独立的显式 reward model，RLHF/DPO 也不是简单地被 pairwise 数据中的最佳样本上界约束，DPO 仍然使用 reference policy。因此选择 **None of the above**。

### 9.2 思考题：最高 lower bound 是否保证低 regret？

课件最后提出：如果总是选择当前的 lower confidence bound 最大的 arm，是否一定有低 regret？答案不能只看保守二字。lower bound 偏向选择已知表现仍然不错的动作，可能让一个真实很好的但不确定的动作长期得不到尝试；是否低 regret 需要具体分析它的探索行为和置信界定义，不能由用了 confidence bound 自动推出。

### 9.3 自测题

1. 为什么 broken-toe 例子是 bandit 而不是有多个决策的 MDP？
2. $\widehat Q_t(a)$ 与 $Q(a)$ 分别是什么类型的量？$N_T(a)\Delta_a$ 在 regret 分解中表示什么？
3. 纯 greedy 为什么可能永久锁定次优动作？
4. 固定 $\varepsilon>0$ 的 $\varepsilon$-greedy 为什么通常有线性 regret？$\varepsilon=0$ 又为什么可能线性 regret？
5. UCB bonus 为什么随 $N_t(a)$ 增加而减小？
6. 集中不等式中的 $\delta$ 与 $\varepsilon$-greedy 中的 $\varepsilon$ 有什么不同？
7. problem-independent 和 problem-dependent regret bound 分别回答什么问题？

<details>
<summary>查看答案</summary>

1. 一位病人只做一次治疗动作并得到结果，没有需要由动作影响的后续状态；若治疗过程中还要连续决策，才更像 MDP。
2. $Q(a)$ 是未知的真实期望，$\widehat Q_t(a)$ 是由历史样本计算的估计；$N_T(a)\Delta_a$ 是选择该动作次数乘以每次相对最优均值的机会损失。
3. 初始样本可能让次优动作估计最高；没有探索机制时，其他动作不再获得样本，估计无法被纠正。
4. 固定 $\varepsilon>0$ 会以正比例持续选择次优动作；$\varepsilon=0$ 退化为可能锁定次优动作的 greedy。
5. 访问少表示不确定性大，较大的 bonus 鼓励获取信息；访问多后均值估计更稳定，bonus 下降。
6. $\delta$ 控制置信界的失败概率，$\varepsilon$ 控制随机探索概率，名称相似但角色不同。
7. 前者给出随总时间的统一最坏情况增长，后者利用具体 gap 和分布相似性刻画实例难度。

</details>

## 10. 本讲小结

- Multi-armed bandit 用未知奖励分布的单步选择抽象探索问题。
- Greedy 只利用当前均值估计，可能被偶然样本锁定。
- Regret 把探索期间相对于最优动作的机会损失写成 gap--count 分解。
- 固定 $\varepsilon$ 的随机探索能避免永久放弃，但会持续支付次优动作的损失。
- Sublinear regret 要求平均机会损失趋近于零；Lai--Robbins 下界说明辨别相似分布仍需要对数级探索。
- UCB 用经验均值加不确定性 bonus 实现 optimism；集中不等式说明 bonus 应随样本数增加而衰减。
- Lecture 10 会修正并展开 UCB regret proof；Bayesian regret 和 Thompson sampling 是本讲目录中的后续方向，但未在本 post deck 中展开。

## Assignment Readiness

- 已覆盖：bandit 建模、动作均值估计、greedy、$\varepsilon$-greedy、regret、gap--count 分解、UCB 直觉和 UCB1 置信半径。
- 还需要：Lecture 10 的 UCB proof、Bayesian bandit / Thompson sampling（若后续作业或课程要求），以及在代码中实现和测试具体 bandit 算法。
- Mastery evidence：目前没有记录独立推导、代码实现或数值实验；覆盖完成不等于已经掌握。
- 推荐下一步：先独立计算一个三臂 bandit 的 $\varepsilon$-greedy 选择概率，再用不同随机种子模拟 greedy 与 UCB 的 regret 曲线。

## 本讲必会公式

### 1. Bandit 目标与动作均值

$$
r_t\sim\mathcal R_{a_t},
\qquad Q(a)=\mathbb E[r\mid a],
\qquad \max\mathbb E\left[\sum_{t=1}^{T}r_t\right].
$$

### 2. Greedy 估计与动作选择

$$
\widehat Q_t(a)=\frac{1}{N_t(a)}\sum_{i=1}^{t}r_i\mathbf 1(a_i=a),
\qquad
a_t=\arg\max_a\widehat Q_{t-1}(a).
$$

### 3. Regret gap--count 分解

$$
L_T=\sum_{a\in\mathcal A}\mathbb E[N_T(a)]\Delta_a,
\qquad
\Delta_a=V^*-Q(a).
$$

### 4. UCB1 结构

$$
a_t=\arg\max_a\left[\widehat Q_{t-1}(a)+
\sqrt{\frac{2\log(1/\delta)}{N_{t-1}(a)}}\right].
$$

完整语义和条件见 §8；这里只作为索引，不重复推导。

## 容易混淆点

1. bandit 的 $Q(a)$ 是单步动作均值；不能无说明地当成 MDP 的 $Q^\pi(s,a)$。
2. regret 用 $V^*-Q(a_t)$ 的期望定义，不是用单次观测的 $V^*-r_t$ 直接替代。
3. $\varepsilon$-greedy 的 $\varepsilon$ 是探索率；UCB 置信界中的 $\delta$ 是失败概率。
4. 固定 $\varepsilon>0$ 与 $\varepsilon=0$ 都可能线性 regret；关键是次优动作是否被线性次数选择。
5. UCB 的 optimism 不是声称动作真的高回报，而是在当前信息下保留其可能高回报的上界。
6. Lecture 9 的 UCB proof 有课后更正说明；不要把不完整的 Lecture 9 草稿当成完整证明。

## 11. 延伸阅读

### 经典基础

- Auer, Cesa-Bianchi, and Fischer, “Finite-time Analysis of the Multiarmed Bandit Problem” (2002)，UCB1 的经典有限时间分析。<https://doi.org/10.1023/A:1013689704352>
- Lai and Robbins, “Asymptotically Efficient Adaptive Allocation Rules” (1985)，渐近 regret 下界的经典来源。<https://doi.org/10.2307/2241181>
- Lattimore and Szepesvári, Bandit Algorithms，课件 UCB proof sketch 直接引用的教材。<https://tor-lattimore.com/downloads/book/book.pdf>

### 前沿动态

截至 2026-08-13 核实：本讲不额外添加前沿算法条目。Lecture 9 的 post deck 目标是建立 bandit regret/UCB 基础，加入未经本地课件或可靠原始来源核验的最新方法会混淆课程主线。
