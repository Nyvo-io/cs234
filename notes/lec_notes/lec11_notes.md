---
title: CS234 Lecture 11 - Bayesian Bandits and Thompson Sampling
aliases:
  - CS234 Lec11
tags:
  - cs234
  - reinforcement-learning
  - multi-armed-bandit
  - bayesian-inference
  - thompson-sampling
---

# CS234 Lecture 11 Notes: Bayesian Bandits and Thompson Sampling

来源：`lecture/lecture11post.pdf`，CS234 Winter 2026，Emma Brunskill；共 50 个物理 PDF 页面。文件名与课程进度对应 Lecture 11，但 PDF 标题页误写为 Lecture 13，且标题页明确标注 `Typo: Lecture 11`。

笔记规范：`cs234-rl-tutor v2`。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-14（经典来源与课件引用核对）。

## 0. 本讲覆盖清单

- [x] 第 1--3 页：标题页的 Lecture 13 / Lecture 11 更正、deterministic bandit 复习题及答案；写入 §1 和 §10。
- [x] 第 4 页：课程 survey 反馈与本讲教学调整；写入 §1。
- [x] 第 5--8 页：课程位置、Bayesian bandits / Thompson sampling / Bayesian regret 目录和 bandit notation recap；写入 §1--§2。
- [x] 第 9--14 页：Bayesian prior、Bayes rule、posterior、conjugacy；写入 §2。
- [x] 第 15--17 页：Bernoulli--Beta 共轭与 Bayesian decision making；写入 §3。
- [x] 第 18--20 页：Bayesian bandit overview 与 Thompson sampling algorithm；写入 §4。
- [x] 第 21--30 页：broken-toe Thompson sampling toy example、posterior updates 和 optimism 对比；写入 §5。
- [x] 第 31--35 页：probability matching 及 Thompson sampling 的等价解释；写入 §6。
- [x] 第 36--38 页：frequentist regret、Bayesian regret 和 optimism upper bound；写入 §7。
- [x] 第 39--41 页：Thompson sampling 的经验/理论边界和 contextual news recommendation；写入 §8。
- [x] 第 42--43 页：新闻推荐场景的理解检查及答案；写入 §8 和 §10。
- [x] 第 44--45 页：Bayesian bandit 的最优策略计算困难与 Gittins index；写入 §9。
- [x] 第 46--50 页：课程能力目标、Bayesian regret bound、PAC/regret toy table；写入 §10。

课件第 33 页是 Thompson sampling 的重复动画页，第 8、19、36、46 页重复目录页；笔记按概念完整性合并，不把动画重复当成新的知识点。PDF 的文字提取会把希腊字母和不等号读成乱码，正文以渲染后的课件公式为准。

## 1. 本讲主线

与 Lecture 10 的关系：Lecture 10 在固定但未知的奖励分布上用 UCB 的置信上界驱动探索；

本讲把“不确定的奖励”进一步表示成一个 posterior distribution，用先验知识和观察数据共同决定下一步行动。这样，探索不再只是一个确定的 bonus，而可以解释为“按照当前 posterior，某个动作成为最优的概率”。

**本讲路线图**

1. 先把 frequentist bandit 的未知参数改写成带 prior 的 Bayesian bandit，并回顾 Bayes rule。
2. 在 Bernoulli reward 下用 Beta 分布形成共轭更新，把 posterior 更新变成成功/失败计数加一。
3. 用 Thompson sampling 从每个 arm 的 posterior 采样一个可能的 reward 参数，再选择采样值最大的 arm。
4. 证明这种抽样规则等价于 probability matching，并比较它与 optimism 的探索行为。
5. 最后讨论 Bayesian regret、contextual recommendation、Gittins index 和 PAC/regret 的边界，不把课件中的结论扩展成未给条件的普遍定理。

课程 survey 的反馈是希望有更清晰的高层结构、概念解释和具体例子；本笔记因此把公式放在问题动机和完整迭代之后解释。

**UCB：给每个 arm 算一个“乐观上界”再选最大的。**  

**Thompson Sampling：给每个 arm 的未知真实均值建立一个 posterior，然后从 posterior 里随机抽一个可能的均值，再选最大的**

上一讲我们有 K 个 arms。

每个 arm ai​ 有一个真实但未知的平均 reward：

Q(ai​)，$Q(a_i)=\mathbb E[r\mid a_i]$ 
反复做动作 a，平均能得到多少 reward

我们通过不断采样，然后得到经验均值：$\hat Q_t(a_i)$

再加一个 confidence bonus：​

$U_t(a_i)=\hat Q_t(a_i)+\sqrt{\frac{2\log t}{N_t(a_i)}}$

这一讲不满足于说：

> “Q(ai​) 我不知道，但它是一个固定常数。”

而是说：

> “既然我不知道 Q(ai​)，那我就用一个**概率分布**描述我目前对它的认识。”


把前两讲里“固定但未知的真实奖励均值 Q(a)”重新写成“由未知环境参数 θ 决定的 Qθ​(a)”；然后开始对这个未知参数 θ 做 Bayesian inference。



## 2. 从未知分布到 Bayesian bandit

与上一讲的关系：Lecture 10 只要求每个 arm 的奖励满足一定范围或集中性质，并通过数据构造置信上界；本节先改变知识表示方式，把未知参数本身当成随机变量，用 posterior 表达“目前认为它可能是什么”。

**本节路线图**

1. 固定 bandit 的 history 与 regret 记号，明确 Bayesian 方法新增的是 prior，而不是新的 reward 定义。
2. 用 Bayes rule 把 prior、likelihood 和观测数据合成为 posterior。
3. 说明一般 posterior 计算可能困难，conjugacy 为什么能让更新保持解析形式。

### 2.1 Bandit 记号复习

*首次完整讲解：Lecture 9 §2.1「形式化定义」与 §4.1「单步 regret 与总 regret」。本节只补充：Bayesian regret 如何在固定参数的 frequentist regret 外再取 prior 平均。*

之前是 $Q(a_i)=\mathbb E[r\mid a_i]$ 

多臂 bandit 仍然由动作集合 $\mathcal A$ 和每个动作的奖励分布 $\mathcal R_a$ 组成。第 $t$ 轮选择 $a_t\in\mathcal A$，然后观察

$$
r_t\sim\mathcal R_{a_t}.
$$

这里 $Q_\theta(a)=\mathbb E[r\mid a,\theta]$ 是 固定环境参数 $\theta$ 下动作 $a$ 的真实期望奖励 （把之前被省略掉的物理环境状态显式的用参数表达了出来）

$a^*_\theta$ 是该环境中的最优动作。前 $T$ 轮的 [frequentist regret](academic-term-lookup:frequentist%20regret) 仍然是(还不是后面的Bayesian regret)

$$
\operatorname{Regret}(A,T;\theta)
=
\mathbb E_{\tau}
\left[
\sum_{t=1}^{T}
\bigl(Q_\theta(a^*_\theta)-Q_\theta(a_t)\bigr)
\middle|\theta
\right],
$$
>	它假设真实 θ 固定，算法在这个固定环境里的 regret 是多少？

其中 $\theta$ 表示固定但未知的环境参数，$\mathbb E_\tau$ 只对算法产生的 action--reward history 取期望。

$\mathbb E_\tau[\cdots\mid\theta]$

意思是：

> **固定环境 θ 以后，对运行算法时可能产生的随机 trajectory/history 取平均。**

为什么 trajectory 还随机？

因为 reward 本身是随机的。

例如：

$r_t\sim\operatorname{Bernoulli}(0.8)$

真实成功率明明是 0.8，但一次具体结果可以是： 1,1,0,1,0,…

算法看到不同 reward，也可能导致后面选择不同的 actions。

所以整个：
$(a_1,r_1,a_2,r_2,\dots)$

是随机的。

因此要取 expectation。



Bayesian 方法不改变这个单次环境中的 regret 定义，而是在此之外再对 $\theta$ 的 prior 取平均。

这个是：
> 固定 θ，算法运行结果有随机性。

Bayesian 方法是：

> Bayesian 意义下，我们还不知道究竟是哪一个 θ，于是对 prior 再平均一次。


### 2.2 Bayesian inference 的对象


Bayesian bandit 给每个未知奖励参数 一个 prior。例如，若 arm $i$ 的奖励分布 reward distribution 由参数 $\phi_i$ 决定，$\phi_i$它就是 arm i 的成功概率，它未知。 则初始不确定性写成 $p(\phi_i)$，它可能是多少。观察数据后，更新为 $p(\phi_i\mid h_t)$，它是 posterior，其中 history 可写成

$$
h_t=(a_1,r_1,\ldots,a_{t-1},r_{t-1}).
$$

prior 不是“算法认为某个参数一定正确”，而是把已有知识或建模假设编码成 参数的概率分布。若 prior 与真实环境严重不匹配，posterior 可能在很长时间内被错误信念牵引；后面的新闻推荐复习题专门检查这个边界。


prior：

>	在看数据之前，我对未知参数有哪些可能性的概率描述

比如有一个二臂老虎机，两个 arm 的成功概率分别是 ： θ1​,θ2​

但我们不知道它们是多少。

在普通 frequentist bandit 里，你可能只是说：

> θ1​,θ2​ 是未知参数，我通过采样估计它们。

而 Bayesian bandit 会更进一步，先写：

$\theta_1 \sim \mathrm{Beta}(1,1)$

$\theta_2 \sim \mathrm{Beta}(1,1)$

这里的意思不是说“真实的 θ1​ 每次都会随机变化”，而是：

> 在还没有观察数据之前，我用 Beta(1,1) 这个分布表示自己对 θ1​ 的不确定性

我认为 θ 可能是 0∼1 之间的任何值，而且目前每个值都差不多一样可信


posterior:

>	观察数据之后，就变成了posterior


$\boxed{\text{prior}\xrightarrow{\text{观察 }r_i}\text{posterior}}$


$\theta=(\phi_1,\phi_2,\dots,\phi_K)$
也就是说：

> θ 是整个环境参数；  
> ϕi​ 是其中第 i 个 arm 的参数。



### 2.3 Bayes rule


对 arm $i$  第一次观察到奖励 $r_{i1}$ 时，Bayes rule 为

$$
p(\phi_i\mid r_{i1})
=
\frac{p(r_{i1}\mid\phi_i)p(\phi_i)}{p(r_{i1})}
=
\frac{p(r_{i1}\mid\phi_i)p(\phi_i)}
{\displaystyle\int p(r_{i1}\mid\phi_i)p(\phi_i)\,d\phi_i}.
$$

整个公式的含义是：posterior 与 likelihood 乘 prior 成正比，再用分母把它归一化为一个概率分布

$\boxed{p(\phi_i)\quad\xrightarrow{\;r_i,\;p(r_i\mid\phi_i)\;}\quad p(\phi_i\mid r_i)}$


$\phi_i$
是 arm i 的**真实 reward 参数**。但不知道真实的它是多少，所以Bayesian 方法不给它直接猜一个确定值，而是给它一个概率分布


$p(\phi_i)$
表示：

>  先验分布，还没有看到这次 reward 以前，我多相信这个参数。
>  
>  比如   $P(\phi_i=0.2)=0.5$
>  有一半几率，\phi_i = 0.2


$p(r_{i1}\mid\phi_i)$ 是likelihood
表示：

> **假如真实参数就是 ϕi​，那我刚刚看到这个 reward r 的概率有多大？**

设这里规定 reward 是 Bernoulli：

$p(r\mid\phi)=\phi^r(1-\phi)^{1-r}$

$\phi_i=0.8$，就是arm i 每拉一次，得到 reward 1 的概率是 0.8，为0的概率是0.2 。
所以：
$P(r_{i1}=1\mid \phi_i=0.8)=0.8$

---

为什么叫做likelihood：

现在 r=1 被观测到了已经：

既然我已经看到了一次成功 r=1，那么不同的 ϕ 对这次观察的解释能力怎么样？”

比如： ϕ=0.1

那么：
$L(0.1)=p(r=1\mid\phi=0.1)=0.1$

说明：
> 如果这个 arm 的真实成功率只有 10%，那么刚刚出现成功不是特别容易发生。

如果：ϕ=0.9

那么：
$L(0.9)=p(r=1\mid\phi=0.9)=0.9$

说明：
> 如果真实成功率是 90%，那刚才看到一次成功就非常合理。

所以观测到一次成功以后：

ϕ=0.9 比 ϕ=0.1 **更能解释我们看到的数据。**

**似然函数（likelihood function）**”这个名字，核心就是：

> **数据已经发生了以后，我们拿这个数据去衡量：不同的参数值 哪个更容易 产生了这份数据。**



$p(\phi_i\mid r_i)$
就是：

> posterior，后验分布。我已经看到 ri​ 了，现在我觉得 ϕi​ 是多少？调整他的分布概率



$\boxed{\text{posterior}\propto\text{likelihood}\times\text{prior}}$
含义是：

> 以前我有一个看法 prior；  
> 新数据告诉我哪些参数更能解释数据 likelihood；  
> 两者结合得到新的看法 posterior。



$p(\phi_i\mid r)=\frac{p(r\mid\phi_i)p(\phi_i)}{p(r)}$

其中：

$p(r)=\int p(r\mid\phi_i)p(\phi_i)\,d\phi_i$

这个分母主要作用就是：

> **把 numerator 归一化，让最终 posterior 的总概率等于 1。**



它把“观察到的数据”转化成对未知奖励参数的新信念；分母只是归一化常数，不需要逐个参数手工比较。这里写的是连续参数的积分形式；若参数只有有限个可能值，分母应改为对这些可能值求和


> [!example] 具体计算：一次二元观测如何改变信念
> 设参数 $\phi$ 只有两个可能值 $\{0.2,0.8\}$，先验分别为 $p(\phi=0.2)=p(\phi=0.8)=0.5$。若观察到成功 $r=1$，且 $p(r=1\mid\phi)=\phi$，则
>
> $$
> p(\phi=0.8\mid r=1)
> =
> \frac{0.8\times0.5}{0.2\times0.5+0.8\times0.5}
> =0.8.
> $$
>
> 一次成功把高成功率参数的 posterior 从 $0.5$ 提高到 $0.8$，但没有把它变成确定事实。若之后继续观察，posterior 会继续由 likelihood 和原有 posterior 共同更新。


==而且 posterior 会成为下一轮的 prior==

这是 Bayesian updating 的连续过程：

第一次之前：
$p(\phi)$

看到 r1​：

$p(\phi\mid r_1)$

第二次之前，就把这个当成新的 prior。

看到 r2​ 后：

$p(\phi\mid r_1,r_2)$

继续：

$p(\phi\mid r_1,r_2,r_3,\dots)$

因此整体过程就是：

$\boxed{\text{prior}\rightarrow\text{data}\rightarrow\text{posterior}\rightarrow\text{new data}\rightarrow\text{new posterior}}$



### 2.4 Conjugate prior

一般情况下，Bayes rule 的积分和归一化可能没有容易计算的闭式解。如果 prior 与 likelihood 组合后，posterior 仍属于同一个参数分布族，就称它们是 **共轭（conjugate）** 的。共轭性是计算上的便利，不是说这个 prior 一定更符合真实世界；指数族通常具有常用的共轭先验。



## 3. Bernoulli--Beta 共轭更新

与第 2 节的关系：第 2 节给出了任意参数的 Bayes update（但第二节我为了有助于理解，直接选定了beta分布和伯努利分布。它本质是描述 抽象关系式，也可以套用进其他的概率分布）；

本节选择 bandit 课件反复使用的二元 reward，把一般积分更新化成可直接计数的 Beta posterior。

**本节路线图**

1. 写出 Bernoulli likelihood 和 Beta density 的角色。
2. 推出单次 reward 以及多次观察后的 posterior 参数更新。
3. 用一个成功/失败序列算出 posterior mean，并说明它如何参与动作选择。

### 3.1 Bernoulli reward 与 Beta prior

*Lecture 9 §2.1「形式化定义」已介绍 Bernoulli bandit 的建模；本节只补充：把成功率当作 Bayesian 参数并用 Beta prior 更新。*

若每次奖励只可能为 $0$ 或 $1$，可以写成

$$
r\mid\theta\sim\operatorname{Bernoulli}(\theta),
\qquad
\theta\in[0,1].
$$

> **如果这个 arm 的真实成功率是 θ，那么每次 reward 按 Bernoulli(θ) 产生。**


$\theta$ 是该 arm 获得奖励 $1$ 的真实概率，也是 Bernoulli reward 的期望。我们刚开始并不知道它，给这个未知的 θ 一个概率分布。对它使用 Beta$(\alpha,\beta)$ prior： 它是连续性概率分布，很适合描述成功率，比例
$\theta\sim\mathrm{Beta}(\alpha,\beta)$
我们对未知成功率 θ 的概率密度：
$$
p(\theta\mid\alpha,\beta)
=
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
\theta^{\alpha-1}(1-\theta)^{\beta-1},
\qquad 0\le\theta\le1.
$$

其中 $\alpha,\beta>0$，$\Gamma(\cdot)$ 是 Gamma function。这个密度描述的是“参数 $\theta$ 的不确定性”，不是一次 reward 的概率质量函数；不要把 Beta distribution 和 Bernoulli distribution 当成同一个对象。

$\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$

它是为了 把整条曲线调整到总面积等于 1

因为一个合法的概率密度必须满足：

$\int_0^1p(\theta)d\theta=1$​


α 相对大：认为成功率偏高。≈成功证据
β 相对大：认为成功率偏低。≈失败证据

$\boxed{\alpha+\beta\text{ 越大}\Rightarrow\text{分布通常越集中，越确定}}$
因为：

$\theta\sim\mathrm{Beta}(\alpha,\beta)$

它的均值是：

$\mu=\mathbb E[\theta]=\frac{\alpha}{\alpha+\beta}$

而方差是：

$$
\operatorname{Var}(\theta)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

把均值代入：
$\boxed{\operatorname{Var}(\theta)=\frac{\mu(1-\mu)}{\alpha+\beta+1}}$

方差越小，就是分布越集中


在 Bernoulli bandit 中 理解更直观：

$\alpha$可以看成和“成功信息”有关，
$\beta$可以看成和“失败信息”有关。

例如先验：$\mathrm{Beta}(1,1)$

然后观察了 10 次，其中：

5 次成功,5 次失败

$\mathrm{Beta}(6,6)$

如果继续观察，100 次里大约一半成功一半失败，那么可能得到：
$\mathrm{Beta}(51,51)$

两个分布都认为成功率大约是：0.5

但是区别在于：

> Beta(6,6)：我根据十来个数据认为大概是 0.5。

> Beta(51,51)：我根据一百多个数据认为大概是 0.5。

显然第二种情况下我们更有把握。


Beta$(1,1)$ 是 $[0,1]$ 上的均匀分布，这个时候无论sita等于多少，$\theta^{1-1}(1-\theta)^{1-1}$都等于1

表示在这个教学例子中没有偏向任何成功率区间。


### 3.2 Posterior update

如果 prior 是 Beta$(\alpha,\beta)$，观察一个 reward $r\in\{0,1\}$ 后，posterior 仍是 Beta 分布：

$$
p(\theta\mid r)
=
\operatorname{Beta}(\alpha+r,\,\beta+1-r).
$$
其实它就是把 **Bernoulli 的 likelihood** 和 **Beta 的 prior** 代进 第二节的Bayes rule，然后整理指数。

Bayes rule：
$p(\theta\mid r)=\frac{p(r\mid\theta)p(\theta)}{p(r)}$
reward 是 Bernoulli：

$r\mid\theta\sim\operatorname{Bernoulli}(\theta)$

而 Bernoulli 的概率质量函数可以统一写成：

$p(r\mid\theta)=\theta^r(1-\theta)^{1-r}$

θ 的 Beta prior：

θ∼Beta(α,β)

那么：

$p(\theta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}$

代入：
$p(r)$

对于已经观察到的 r 来说，是一个**不依赖 θ 的常数**。

所以研究 posterior 关于 θ 的形状时可以写：
$p(\theta\mid r)\propto p(r\mid\theta)p(\theta)$

现在把刚才两个东西放进去：

$p(\theta\mid r)\propto\underbrace{\theta^r(1-\theta)^{1-r}}_{\text{likelihood}}\underbrace{\theta^{\alpha-1}(1-\theta)^{\beta-1}}_{\text{prior}}$

同底数相乘，指数相加：
$\theta^r\theta^{\alpha-1}=\theta^{\alpha+r-1}$

另一边：
$(1-\theta)^{1-r}(1-\theta)^{\beta-1}=(1-\theta)^{\beta-r}$

所以：
$p(\theta\mid r)\propto\theta^{\alpha+r-1}(1-\theta)^{\beta-r}$

再和 Beta 分布的标准形式比较：

$\operatorname{Beta}(a,b)\quad\Longrightarrow\quad p(\theta)\propto\theta^{a-1}(1-\theta)^{b-1}$

令
$a=\alpha+r$
$b=\beta+1-r$

就能得出上面的式子



一次成功 $r=1$ 只把 $\alpha$ 加一；一次失败 $r=0$ 只把 $\beta$ 加一。观察 $s$ 次成功和 $f$ 次失败后，更新为

$$
p(\theta\mid D)
=
\operatorname{Beta}(\alpha+s,\,\beta+f).
$$

posterior mean 为

$$
\mathbb E[\theta\mid D]
=
\frac{\alpha+s}{\alpha+\beta+s+f}.
$$

这个均值是当前 posterior 下对成功率的平均判断；Thompson sampling 不只使用这个均值，而是从整个 posterior 采样，因此仍会保留不确定性带来的探索。

> [!example] 具体计算：Beta--Bernoulli 更新
> 初始 prior 为 Beta$(1,1)$。某个 arm 依次得到奖励 $0,1,1$，因此 $s=2,f=1$。
>
> $$
> p(\theta\mid D)=\operatorname{Beta}(1+2,1+1)=\operatorname{Beta}(3,2),
> \qquad
> \mathbb E[\theta\mid D]=\frac35=0.6.
> $$
>
> 这里 $0.6$ 是 posterior mean，不是说该 arm 的真实成功率已经被证明等于 $0.6$。后续 Thompson iteration 仍可能从 Beta$(3,2)$ 采到高于或低于 $0.6$ 的候选参数。


### 3.3 Bayesian inference 如何服务决策

Bayesian bandit 的决策链可以压缩成：维护每个 arm 的 reward-parameter posterior，使用 posterior 产生动作选择，观察新 reward，再只更新受到影响的 posterior。若 prior 知识较准确，posterior 能更快集中；若 prior 很误导，过强的先验也会延迟纠正。


---

第 2 节不管 reward 到底是什么分布，它先讲一个**通用框架**：

有一个未知参数：$\phi$

先给它一个 prior：$p(\phi)$

观察数据 r 后，通过 Bayes rule：​

$p(\phi\mid r)=\frac{p(r\mid\phi)p(\phi)}{p(r)}$

得到 posterior： $p(\phi\mid r)$

这个时候它还没有限定：

- reward 是 Bernoulli 还是 Gaussian；
- prior 是 Beta 还是 Gaussian；
- 参数到底代表成功率还是均值。

我之前在第二节写的那些参数分布，都是为了有助于理解，直接设定的。它通常是是 Bayesian 建模时人为选择来表示不确定性

第 3 节说：

> 好，现在我们不要抽象讲了，直接来看 bandit 最常见的情况：reward 只有 0/1。

也就是：

r∈{0,1}

那么 reward 使用 Bernoulli：
$r\mid\theta\sim\operatorname{Bernoulli}(\theta)$

然后专门选择 Beta 作为 prior：

$\theta\sim\operatorname{Beta}(\alpha,\beta)$

于是把第 2 节的一般 Bayes rule：

$p(\theta\mid r)\propto p(r\mid\theta)p(\theta)$

具体算出来以后，发现：

$p(\theta\mid r)=\operatorname{Beta}(\alpha+r,\beta+1-r)$




## 4. Bayesian bandits 与 Thompson sampling

与第 3 节的关系：Beta--Bernoulli 只解决“如何更新一个 arm 的 posterior”；本节把所有 arm 的 posterior 放进一个探索循环，得到 Thompson sampling。

**本节路线图**

1. 说明 Bayesian bandit 相对于 UCB 新增的输入是 prior/posterior。
2. 用四步数据流描述 posterior sampling 的动作选择。
3. 在 Bernoulli 场景中把一般 reward-distribution 版本落成 Beta 参数采样。

### 4.1 Bayesian bandit 的输入与输出

在 Bayesian bandit 中，我们维护的是

$$
p(\mathcal R_a\mid h_t)
$$

或其参数化表示，而不是只保留一个点估计。posterior 可以用于 Bayesian UCB，也可以用于 probability matching；本讲主要展开后者。它带来的优点是能利用已有 prior，代价是结果依赖 prior 的合理性和 posterior 计算的可行性。


### 4.2 Thompson sampling 的算法

Thompson sampling（也叫 posterior sampling）把“动作可能是最优的概率”变成一次随机抽样。每一轮的输入是各 arm 的 posterior，输出是一个动作和新 reward 后的 posterior。

算法流程如下：

1. **初始化**：为每个 arm $a$ 设定 prior $p(\mathcal R_a)$。
2. **独立抽样**：对每个 arm，从当前 posterior 抽取一个可能的奖励分布 $\widetilde{\mathcal R}_a$。
3. **计算候选价值**：令 $\widetilde Q(a)=\mathbb E[\widetilde{\mathcal R}_a]$；Bernoulli 场景中它就是抽到的候选成功率 $\widetilde\theta_a$。
4. **利用候选世界**：选择 $a_t\in\arg\max_a\widetilde Q(a)$。
5. **观察与更新**：观察 $r_t$，只用该 arm 的新数据按 Bayes rule 更新 posterior。
6. **重复**：下一轮重新从更新后的 posterior 抽样，而不是沿用上一轮的候选参数。

数据流：

$$
\text{posterior}
\longrightarrow
\text{sampled candidate world}
\longrightarrow
\text{greedy action in that world}
\longrightarrow
\text{reward}
\longrightarrow
\text{posterior update}.
$$

![[lec11-thompson-algorithm-p20.png|900]]

*图：Lecture 11 物理 PDF 第 20 页的 Thompson sampling 算法页；来源：`lecture/lecture11post.pdf` 第 20 页。这里保留课件原始算法框架，正文对每一步作了符号澄清。*

课件第 20 页的简短算法写作 $Q(a)=\mathbb E[R_a]$；这里加波浪号是为了区分从 posterior 抽到的候选分布和真实未知分布。

## 5. Broken-toe 例子：一次完整 Thompson iteration

与第 4 节的关系：算法流程已经说明“抽样--选择--观测--更新”；本节沿用 Lecture 9 的 broken-toe 教学例子，完整走一轮并与 optimism 的选择序列对比。

**本节路线图**

1. 设三个 arm 的真实 Bernoulli 参数未知，并为每个 arm 设 Beta$(1,1)$ prior。
2. 展示一次 posterior sampling 如何选择 arm $a_3$。
3. 观察失败后把 $a_3$ 的 posterior 更新为 Beta$(1,2)$。
4. 解释之后为什么 $a_3$ 仍可能被再次抽到，以及这和确定性 optimism 的区别。

### 5.1 课件设定

*Lecture 9 §2.1「形式化定义」已介绍 bandit 的动作与奖励分布；本节沿用 §2.2 的 broken-toe 说明例子，只补充：同一教学数据在 Thompson sampling 下如何更新 posterior。*

课件使用三个说明性动作：手术 $a_1$、buddy taping $a_2$、不处理 $a_3$，真实但未知的成功率设为

$$
\theta_1=0.95,
\qquad
\theta_2=0.90,
\qquad
\theta_3=0.10.
$$

这些数字是课件明确标注的虚构教学数据，不是实际医疗疗效。对每个 $\theta_i$ 使用 Beta$(1,1)$ prior。

### 5.2 一轮抽样与更新

从三个 prior 分别抽到候选参数

$$
\widetilde\theta_1=0.3,
\qquad
\widetilde\theta_2=0.5,
\qquad
\widetilde\theta_3=0.6.
$$

最大候选值是 $\widetilde\theta_3$，所以选择 $a_3$。若观察到结果 $r_t=0$，则只有 $a_3$ 的 posterior 更新：

$$
\operatorname{Beta}(1,1)
\xrightarrow{r_t=0}
\operatorname{Beta}(1,2),
\qquad
\mathbb E[\theta_3\mid D]=\frac13.
$$

这一步体现了 Thompson sampling 的核心：选择依据的是一个可能的世界，更新依据的是真实观察到的 reward。候选参数 $0.6$ 不会被直接当成观测值写入 posterior；真正写入的是失败证据 $r_t=0$。

> [!example] 反例对比：为什么一次失败不会永久锁定 arm
> 假设下一轮从 posterior 抽到 $\widetilde\theta_1=0.7$、$\widetilde\theta_2=0.4$、$\widetilde\theta_3=0.2$，则选择 $a_1$。再下一轮仍可能从 $\operatorname{Beta}(1,2)$ 抽到较高的 $\widetilde\theta_3$，因此 $a_3$ 还有机会被重新检查。
>
> 与 Lecture 10 的 UCB 不同，Thompson sampling 的随机性来自 posterior sampling；它不是把一个确定性的 bonus 加到均值上，而是在每轮抽取一个候选环境。

### 5.3 与 optimism 的课件对照

课件 toy table 给出的前五次选择序列是：

| 方法 | 前五次选择 |
|---|---|
| Optimism | $a_1,a_2,a_3,a_1,a_2$ |
| Thompson sampling | $a_3,a_1,a_1,a_1,a_1$ |

这不是两个算法在所有随机运行中的固定序列，而是课件展示的一次 toy run。它说明 optimism 会按照置信上界的确定性规则安排探索，而 Thompson sampling 会因为每轮 posterior 抽样而产生不同的探索顺序。

## 6. Probability matching：Thompson sampling 在做什么

与第 5 节的关系：前一节从一个候选参数样本出发描述算法；本节把这个随机步骤改写成策略概率，解释为什么它不是无目的随机探索。

**本节路线图**

1. 定义动作成为当前最优动作的 posterior probability。
2. 证明从每个 arm 的 posterior 抽样并取最大值，正好实现这个概率。
3. 说明不确定性大的动作可能拥有更高的“成为最优”概率，但不保证每次都被选。

### 6.1 Probability matching 定义

给定 history $h_t$，probability matching 按动作是最优的 posterior probability 选择动作：

$$
\pi(a\mid h_t)
=
\mathbb P
\left[
Q(a)>Q(a')\ \text{for all }a'\ne a
\mid h_t
\right].
$$

它的整体含义是：如果当前 posterior 认为 $a$ 在可能的真实环境中成为最优臂的概率为 $0.3$，那么策略就以 $0.3$ 的概率选择它。这个定义直接把“探索”与当前不确定性联系起来；posterior 越分散，动作成为最大值的概率可能越高。

### 6.2 Thompson sampling 的等价性

令 $R$ 表示从联合 posterior 中抽取的一组候选 reward distributions。对每个 arm 取候选期望并选择最大者，则

$$
\begin{aligned}
\pi(a\mid h_t)
&=\mathbb P\left[Q(a)>Q(a')\ \forall a'\ne a\mid h_t\right]\\
&=\mathbb E_{R\mid h_t}
\left[
\mathbf 1\left(a=\arg\max_{a'\in\mathcal A}Q_R(a')\right)
\right].
\end{aligned}
$$

右侧只是“抽样后 $a$ 获胜”的指示变量期望，也就是 $a$ 获胜概率。因此 Thompson sampling 不是先显式计算一个复杂的多维概率再采样，而是用一次 posterior draw 获得一个服从该概率的 action。

课件还指出，probability matching 常表现出 optimism under uncertainty：不确定动作的 posterior 仍有一部分质量落在很高的 reward 区域，所以它们可能有较高的成为最大值概率。这是“可能有利”的探索机制，不是对未知动作高回报的保证。

## 7. Frequentist regret 与 Bayesian regret

与第 6 节的关系：probability matching 描述动作如何产生；本节回到评价问题，区分“固定真实参数下评估算法”和“从 prior 平均多个可能环境”。

**本节路线图**

1. 写出 frequentist regret，并明确条件在固定真实参数 $\theta$ 上。
2. 对 $\theta$ 的 prior 再取期望，得到 Bayesian regret。
3. 用 optimism event 给出 regret 上界的直觉，并说明课件没有在此页展开完整证明。

### 7.1 两种 regret

*首次完整讲解：Lecture 9 §4.1「单步 regret 与总 regret」。本节只补充：Bayesian regret 在固定参数条件外再对环境 prior 取期望。*

Frequentist regret 假设存在一组固定但未知的参数 $\theta$：

$$
\operatorname{Regret}(A,T;\theta)
=
\mathbb E_{\tau}
\left[
\sum_{t=1}^{T}
\bigl(Q_\theta(a^*_\theta)-Q_\theta(a_t)\bigr)
\middle|\theta
\right].
$$

Bayesian regret 则再对参数 prior 取平均：

$$
\operatorname{BayesRegret}(A,T)
=
\mathbb E_{\theta\sim p(\theta)}
\mathbb E_{\tau\mid\theta}
\left[
\sum_{t=1}^{T}
\bigl(Q_\theta(a^*_\theta)-Q_\theta(a_t)\bigr)
\right].
$$

第一个公式问“在这个固定环境中算法损失多少”；第二个公式问“按先验平均后，算法损失多少”。Bayesian regret 不是把每轮的 reward 随便再平均一次，而是把环境参数的不确定性也纳入期望。

> [!example] 具体计算：同一动作序列下的两种 regret
> 设两臂的真实均值为 $Q_\theta(a_1)=0.8$、$Q_\theta(a_2)=0.5$，算法两轮都选择 $a_2$，因此固定环境下的 regret 是
>
> $$
> \operatorname{Regret}(A,2;\theta)=2(0.8-0.5)=0.6.
> $$
>
> 若 prior 以 $0.5$ 概率认为环境为 $(0.8,0.5)$，以 $0.5$ 概率认为环境为 $(0.6,0.5)$，而算法在两种环境中都两轮选择 $a_2$，则 Bayesian regret 为
>
> $$
> \operatorname{BayesRegret}(A,2)
> =0.5\times0.6+0.5\times2(0.6-0.5)=0.4.
> $$
>
> 前者固定环境参数后评价 history 随机性，后者还把“哪个环境是真的”按 prior 加权。

### 7.2 Optimism upper bound 的桥梁

若某个事件上每轮的上置信界满足 $U_t(a^*)\ge Q(a^*)$，且算法选择 $a_t$ 最大化 $U_t$，则

$$
Q(a^*)-Q(a_t)
\le
U_t(a_t)-Q(a_t).
$$

把逐轮不等式求和并取期望，就得到

$$
\operatorname{Regret}(A,T;\theta)
\le
\mathbb E_\tau
\left[
\sum_{t=1}^{T}
\bigl(U_t(a_t)-Q(a_t)\bigr)
\middle|\theta
\right]
$$

在该 upper-bound 事件上成立。它解释了 Lecture 10 的 proof 为什么关注 bonus，但这页课件没有给出 Thompson sampling 的完整 regret theorem；不要把这个不等式本身当作 TS 的最终 bound。

### 7.3 课件对 Thompson sampling 理论状态的表述

课件明确提醒：截至课件制作时，标准 Thompson sampling 的 frequentist bounds 不匹配 frequentist algorithms 的最佳 bounds；另一方面，经验上它尤其适用于 contextual multi-armed bandit。这里的“经验上有效”不是一个适用于所有 prior、模型和数据分布的定理。

## 8. Contextual bandit：新闻推荐例子

与第 7 节的关系：前面的 regret 记号把 $Q(a)$ 写成不随上下文变化的动作均值；contextual bandit 在每轮先看到 context，context 会改变各 arm 的 reward，因此更接近新闻推荐等实际问题。

**本节路线图**

1. 区分 context、article arm 和 click reward。
2. 解释 delayed feedback 下为什么 posterior sampling 的随机性有实用价值。
3. 用理解检查明确 Thompson sampling 的优势与 misleading prior 的风险。

### 8.1 News article recommendation

课件引用 Chapelle and Li (2010) 的新闻推荐场景：每轮 context 影响文章的 reward，context 假设按步独立同分布采样；arms 是文章，reward 是用户点击（记为 $+1$），$Q(a)$ 可理解为点击率。这里的上下文使“某篇文章的平均点击率”依赖当前访问者或环境，不能直接套用无 context 的 tabular UCB 公式。

课件还强调 delayed feedback 的实际问题：网站可能在前一个用户是否点击尚未返回时，已经迎来下一位用户。Thompson sampling 的 posterior sampling 可以在反馈到达前为不同用户产生随机化的展示决策；它不是因为 optimism 算法在数学上“不能工作”，而是避免一个确定性选择器在没有新反馈时反复选同一动作。

### 8.2 理解检查

课件的三条判断中，正确答案是 1 和 3：

1. **正确**：在 delayed feedback 和高并发场景中，Thompson sampling 的随机化可以避免未收到新反馈时始终展示同一个 action。
2. **错误**：课件没有因此声称 optimism 在该 setting 有更强的 regret bound；前一页反而说明标准 TS 的 frequentist bound 与最佳 frequentist bound 的关系并不简单。
3. **正确**：misleading prior 可能让 TS 长时间偏向错误动作。

课件给出的风险例子是：某个真实 Bernoulli 参数为 $0.1$ 的 arm，却使用 Beta$(100,1)$ prior。这个 prior 对高 $\theta$ 赋予很大权重，因此需要较多失败数据才能显著拉低 posterior。这个例子说明 prior 是模型的一部分，不能只因为 Bayesian 方法“利用先验”就假设结果自动更好。

## 9. Bayesian bandit 的最优策略与 Gittins index

与第 8 节的关系：Thompson sampling 是一种计算便宜、随机化的 posterior policy，但“效果好”不等于“对已知 prior 和 horizon 严格最优”。本节记录课件给出的更高层次边界。

**本节路线图**

1. 说明已知 prior 和 horizon 时，原则上可以求最大化有限期望 reward 的策略。
2. 说明直接把 history 映射到下一动作会带来计算困难。
3. 介绍 index policy 和 Gittins index 的课程级定义，但不展开其动态规划推导。

### 9.1 最优策略的计算困难

给定 prior 和已知 horizon，理论上可以寻找最大化可用 horizon 内期望累计 reward 的决策 policy。然而，朴素表示会让 policy 依赖完整 history：已经拉过哪些 arm、观察到哪些 reward、当前各 arm 的 posterior 是什么。history 空间随时间快速增长，因此直接枚举 policy 通常不可行。

### 9.2 Index policy 与 Gittins index

课件把 **index policy** 定义为：为每个 arm 计算一个实值 index，只使用该 arm 的统计量和 horizon，然后选择 index 最大的 arm。对 Bayesian multi-armed bandit，**Gittins index** 是在折扣累计 reward 目标下的最优策略 index。这个结论有明确的目标条件：课件说的是 Bayesian bandit 的 expected discounted reward，不是任意 finite-horizon 或任意 contextual bandit 的通用最优性。

本讲只需要知道它解决的结构性问题：把依赖全局 history 的决策压缩成每个 arm 一个 index。Gittins index 的具体计算和证明不在本 post deck 中展开。

## 10. 课程检查、PAC 与本讲边界

与前面各节的关系：本节把课件最后的理解目标和两个易混点集中整理，帮助区分已讲的算法、课件给出的理论结论和未展开的 PAC 细节。

**本节路线图**

1. 先回答 deterministic bandit 与 UCB 的开头复习题。
2. 解释 Bayesian regret bound 和课件的 PAC/regret toy table。
3. 汇总本讲可操作的能力目标、未覆盖内容和自测题。

### 10.1 Deterministic bandit 复习题

课件答案是：**UCB will have sublinear regret with probability one**。在 deterministic reward 环境中，一个 arm 的单次观测已经等于它的真实期望奖励，因此置信上界不会因为 reward noise 而失败。完成必要的初始化后，算法可以识别最优 arm；若存在并列最优臂，选择它们也不会产生 regret。这里的结论依赖 deterministic reward 和正确初始化，不能推广为“任何非随机或有噪声环境都自动满足同样结论”。

### 10.2 Bayesian regret bound 的课件结论

课件第 49 页给出 Bayesian regret 的表达，并用一句总结性文字说 posterior sampling 在忽略常数时具有与 UCB 相同的 regret bounds。由于该页没有给出算法、prior、reward family 和 horizon 的完整适用条件，笔记只把它记录为**课件结论性摘要**，不把它改写成无条件 theorem，也不声称标准 TS 在所有 frequentist 评价下都达到 UCB 或最优界。

### 10.3 PAC 与 regret 的 toy 对照

课件最后用三个 broken-toe arm、$\epsilon=0.05$ 和前五步的 Optimism/TS 序列比较 regret 与“within $\epsilon$”指标。这里的 PAC 直观指标可以写为

![[lec11-pac-regret-p50.png|900]]

*图：Lecture 11 物理 PDF 第 50 页的 PAC/regret toy table；来源：`lecture/lecture11post.pdf` 第 50 页。图中 `W/in \epsilon` 表示 within-$\epsilon$。*

$$
\operatorname{Within}_\epsilon(t)
=
\mathbf 1\left(
Q(a_t)\ge Q(a^*)-\epsilon
\right).
$$

它检查当前动作是否已经足够接近最优动作；它与 regret 不同：regret 衡量差距大小，within-$\epsilon$ 只做合格/不合格判断。

在课件表格中，$Q(a_1)=0.95$、$Q(a_2)=0.90$、$Q(a_3)=0.10$，$\epsilon=0.05$。因此 $a_1$ 和 $a_2$ 都 within $\epsilon$，$a_3$ 不满足。表中 Thompson sampling 的第一步选 $a_3$，产生 $0.85$ regret 且不 within $\epsilon$；随后四步选择 $a_1$，regret 为 $0$ 且 within $\epsilon$。这个小表说明一个算法可能在大部分时间已经近似正确，但仍有一次明显探索损失；PAC 风格的合格率和 regret 曲线提供的是不同视角。

### 10.4 What you should understand

本讲结束后，课件要求能够：

- 解释 bandit 与 MDP 的关系，并定义 regret 和 PAC 风格的 within-$\epsilon$ 指标；
- 说明 UCB 为何具有 sublinear regret，并举例解释 greedy、固定 $\varepsilon$-greedy 和 pessimism 可能产生 linear regret；
- 实现 UCB bandit；
- 为 Bernoulli reward 实现 Thompson sampling。

其中前三项的理论背景来自 Lecture 9--11；“实现”是能力目标，不等于笔记已经验证过代码。

### 10.5 自测题

1. prior、likelihood 和 posterior 分别代表什么？
2. 为什么 Beta$(\alpha,\beta)$ 是 Bernoulli likelihood 的共轭 prior？
3. prior Beta$(1,1)$ 观察 $0,1,1$ 后，posterior 和 posterior mean 是什么？
4. Thompson sampling 的候选参数为什么不能当成真实观测直接写入 posterior？
5. probability matching 的策略概率与 Thompson sampling 的 posterior draw 如何相等？
6. frequentist regret 与 Bayesian regret 的期望对象有什么不同？
7. misleading prior 为什么可能让 Thompson sampling 表现很差？
8. Gittins index 的 optimality 依赖哪些目标条件？
9. within-$\epsilon$ 与 regret 分别衡量什么？

<details>
<summary>查看答案</summary>

1. prior 是观察数据前对未知参数的分布，likelihood 是给定参数时数据出现的概率，posterior 是二者结合后的更新分布。
2. likelihood 与 Beta density 相乘后，关于 $\theta$ 的幂次仍是 Beta 形式，所以 posterior 只需更新两个计数参数。
3. posterior 是 Beta$(3,2)$，posterior mean 是 $3/5=0.6$。
4. 候选参数只是从当前信念抽出的假设；真正能更新 posterior 的是环境返回的 reward。
5. 每次 posterior draw 中动作 $a$ 成为最大候选价值的指示变量，取期望正好得到它成为最优动作的 posterior probability。
6. frequentist regret 固定 $\theta$ 后只对 history 取期望；Bayesian regret 还对 $\theta\sim p(\theta)$ 取期望。
7. 强先验可能给低真实成功率的 arm 赋予高参数概率，需要很多反例数据才能纠正。
8. 课件给出的条件是 Bayesian multi-armed bandit、以 expected discounted reward 为目标；不能无条件推广到任意 horizon 或 contextual bandit。
9. regret 保留性能差距的数值大小；within-$\epsilon$ 只判断动作是否达到指定近似门槛。

</details>

## 本讲小结

- Bayesian bandit 用 prior 和 posterior 表达对未知 reward distribution 的不确定性。
- Bayes rule 是 posterior 更新的总公式；conjugacy 让 Beta--Bernoulli 更新变成成功/失败计数。
- Thompson sampling 每轮从各 arm posterior 抽候选 reward 参数，再选择候选最优动作并用真实 reward 更新。
- Probability matching 说明 TS 的 action probability 等于该动作成为 posterior 最优动作的概率。
- Bayesian regret 是对环境参数 prior 平均后的 regret；它与固定环境下的 frequentist regret 不是同一个期望。
- Contextual news recommendation 体现了 delayed feedback、随机化展示和 misleading prior 的实际边界。
- Gittins index 是特定 Bayesian discounted bandit 目标下的 index-policy 结果，不是本讲展开的通用算法。
- PAC 风格 within-$\epsilon$ 指标和 regret 分别衡量“是否足够接近最优”和“差多少”。

## Assignment Readiness

- 已覆盖：Bayesian bandit 建模、Bayes rule、Beta--Bernoulli 共轭、Thompson sampling、probability matching、frequentist/Bayesian regret、contextual bandit 边界和 Gittins index 的课程级定位。
- 可以开始：若后续作业要求实现 Bernoulli Thompson sampling，理论前置已经覆盖；仍需根据真实 starter code 完成接口映射、运行测试和 regret 实验。
- 尚未覆盖：PAC 的完整样本复杂度定理、Bayesian UCB 的具体算法与证明、Gittins index 的计算方法和证明、contextual Thompson sampling 的具体模型实现。
- Mastery evidence：没有记录独立 Bayes 推导、Thompson sampling 实现、模拟结果或 Bayesian regret 证明；coverage 不等于 mastery。
- 推荐下一步：先手算 Beta$(1,1)$ 在 $0,1,1$ 后的 posterior，再用三个 Bernoulli arm 实现一次“sample--argmax--observe--update”循环。

## 本讲必会公式

### 1. Bayes rule

$$
p(\phi\mid D)=\frac{p(D\mid\phi)p(\phi)}{p(D)}.
$$

完整解释见 §2.3。

### 2. Beta--Bernoulli posterior

$$
\operatorname{Beta}(\alpha,\beta)
\xrightarrow{s\ \text{successes},\ f\ \text{failures}}
\operatorname{Beta}(\alpha+s,\beta+f).
$$

完整解释见 §3.2。

### 3. Thompson sampling action rule

$$
\widetilde\theta_a\sim p(\theta_a\mid h_t),
\qquad
a_t\in\arg\max_{a\in\mathcal A}\widetilde\theta_a
$$

Bernoulli 场景的完整循环见 §4.2 和 §5.2。

### 4. Probability matching

$$
\pi(a\mid h_t)
=
\mathbb P\left[Q(a)>Q(a')\ \forall a'\ne a\mid h_t\right].
$$

完整解释见 §6。

### 5. Bayesian regret

$$
\operatorname{BayesRegret}(A,T)
=
\mathbb E_{\theta\sim p(\theta)}
\mathbb E_{\tau\mid\theta}
\left[
\sum_{t=1}^{T}
\bigl(Q_\theta(a^*_\theta)-Q_\theta(a_t)\bigr)
\right].
$$

完整解释见 §7.1。

## 容易混淆点

1. Beta 分布是参数 $\theta$ 的 distribution，Bernoulli 分布是 reward $r$ 在给定 $\theta$ 下的 distribution。
2. Thompson 的 sampled parameter 是候选假设，不是环境真实参数，也不是新的 reward observation。
3. posterior mean 不等于 Thompson sampling 的 action value；TS 使用整个 posterior 的随机抽样。
4. Bayesian regret 对 prior 平均，frequentist regret 固定环境参数；两者的评价问题不同。
5. Probability matching 的随机性有 posterior 解释，不是均匀随机探索。
6. misleading prior 可以严重拖慢学习；“利用 prior”不是无条件收益。
7. 课件的 TS regret 摘要没有完整定理条件；不能把“忽略常数同阶”当成所有设置下的严格结论。
8. Gittins index 的 optimality 针对 Bayesian discounted multi-armed bandit，不自动适用于 contextual 或任意 finite-horizon setting。
9. PAC within-$\epsilon$ 只给合格/不合格，不能替代 regret 对损失大小的刻画。

## 延伸阅读

### 经典基础

- Lattimore and Szepesvári, *Bandit Algorithms*，Bayesian bandit、Thompson sampling 和 Gittins index 的教材背景。<https://tor-lattimore.com/downloads/book/book.pdf>
- Chapelle and Li, “An Empirical Evaluation of Thompson Sampling,” NeurIPS 2011；课件新闻推荐案例直接引用 Chapelle and Li (2010) 的工作线索。<https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling>
- Gittins, *Multi-Armed Bandit Allocation Indices* (1989)，Gittins index 的经典来源。<https://doi.org/10.1007/978-1-4612-3618-5>

### 前沿动态

截至 2026-08-14 核实：本讲不额外添加前沿条目。课件的重点是 Bayesian bandit 和 Thompson sampling 的基础机制与边界；加入未经本地课程需要核验的最新变体会削弱主线。
