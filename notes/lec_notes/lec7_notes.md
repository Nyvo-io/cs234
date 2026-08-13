---
title: CS234 Lecture 7 - Policy Gradients and Imitation Learning
aliases:
  - CS234 Lec7
tags:
  - cs234
  - reinforcement-learning
  - policy-gradient
  - imitation-learning
  - inverse-rl
---

# CS234 Lecture 7 Notes: Policy Gradients and Imitation Learning

来源：`lecture/lecture7post.pdf`，CS234 Winter 2026，Emma Brunskill，共 72 个物理 PDF 页面。

笔记规范：`cs234-rl-tutor v2`。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-03（延伸阅读只列与本讲直接相关的经典来源；未加入未经核实的当期结论）。

## 0. 本讲覆盖清单

- [x] 第 1 页：标题和课程主题；写入 §1。
- [x] 第 2--4 页：PPO clipped objective 回顾与正负 advantage 的图形判断；写入 §2.1，并回引 Lecture 6。
- [x] 第 5--6 页：课程结构与本讲目标；写入 §1。
- [x] 第 7--9 页：policy-gradient 限制、PPO 两种变体和 advantage 估计问题；写入 §2.1。
- [x] 第 10--16 页：N-step advantage、TD residual、GAE、bias--variance 平衡和 PPO 截断 GAE；写入 §2.2--§2.4。
- [x] 第 17 页：单调改进理论章节过渡；写入 §3。
- [x] 第 18--22 页：performance bound、majorize--maximize 观点、改进保证和 trust region 动机；写入 §3.1--§3.3。
- [x] 第 23--24 页：PPO 与 policy-gradient 总结；写入 §3.3。
- [x] 第 25 页：模仿学习章节过渡；写入 §4。
- [x] 第 26--30 页：从示范学习、reward shaping、应用例子、问题设置；写入 §4.1。
- [x] 第 31--34 页：behavioral cloning、ALVINN、learning to fly 和 BCRNN 经验；写入 §4.2。
- [x] 第 35 页：DAGGER 章节过渡；写入 §4.4。
- [x] 第 36--38 页：behavior cloning 的 compounding errors、训练/测试状态分布错配和二次误差直觉；写入 §4.3。
- [x] 第 39 页：DAGGER dataset aggregation 算法、效果和限制问题；写入 §4.4。
- [x] 第 40 页：reward learning 章节过渡；写入 §5。
- [x] 第 41--43 页：feature-based reward inverse RL 设置、专家最优性和 reward 非唯一性；写入 §5.1。
- [x] 第 44--49 页：线性特征奖励、discounted feature frequency、最优性关系、feature matching 和 ambiguity；写入 §5.2--§5.3。
- [x] 第 50 页：MaxEnt IRL 与 GAIL 方向指引；写入 §5.4 和延伸阅读。
- [x] 第 51 页：MaxEnt IRL 章节过渡；写入 §5.4。
- [x] 第 52--59 页：轨迹 feature counts、路径分布、最大熵原理、指数族、随机 MDP、likelihood gradient、前后向频次算法和 transition-model 需求；写入 §5.4--§5.6。
- [x] 第 60--62 页：从 IRL 回到 policy、课程总结和必会内容；写入 §5.7、§4.2、§10。
- [x] 第 63 页：off-policy policy-gradient 章节过渡；写入 §6。
- [x] 第 64--66 页：importance sampling 定义、估计式和方差；写入 §6.1。
- [x] 第 67--69 页：off-policy policy-gradient 的 prefix ratio、权重连乘和 exploding/vanishing 问题；写入 §6.2。
- [x] 第 70 页：advanced policy-gradient 与 PPO 总结；写入 §6.3。
- [x] 第 71--72 页：REINFORCE 理解检查；写入 §10。

视觉材料决策：第 37 页的状态分布错配、第 39 页的 DAGGER 流程、第 53 页的确定性路径分布和第 58 页的 MaxEnt IRL 前后向频次算法直接嵌入。第 3--4 页的 PPO 图与 Lecture 6 第 48 页重复，因此在文字中回引；公式动画页用可搜索的 Markdown 公式重写。

## 1. 本讲主线

Lecture 6 已经说明 PPO 为什么用 clipped ratio 或 KL penalty 限制策略更新。

本讲先回答一个更具体的问题：PPO 里的 advantage 到底怎样从 rollout 估计出来；然后给出单调改进理论，解释保守更新为什么有理论动机。后半讲把“通过 reward 学策略”换成“从好行为本身学习”，依次介绍 behavior cloning、DAGGER、feature-based inverse RL 和 maximum-entropy IRL，最后补上 off-policy policy-gradient 的 importance-sampling 代价。

可以把整讲压缩成四条连接：

1. **估计器**：N 步优势估计（N-step advantage）把真实奖励和 value bootstrap 混合，广义优势估计（Generalized Advantage Estimation, GAE）再对不同步长指数加权，在 bias 和 variance 之间调节。
2. **更新理论**：performance bound 把真实性能差下界为 surrogate improvement 减去 KL 变化代价，于是 trust-region / PPO 的保守更新有了来源。
3. **示范学习**：行为克隆（behavioral cloning）直接拟合专家动作；DAGGER 让专家在学习策略自己访问到的状态上继续标注，修复分布错配。
4. **奖励学习与离策略**：IRL 从示范反推 reward，但 reward 和 policy 都不唯一；异策略策略梯度（off-policy policy gradient）可以重用旧数据，却会面对轨迹概率比连乘造成的高方差。

## 2. 从 N-step advantage 到 GAE

### 2.1 为什么 PPO 需要更好的 advantage estimate

*首次完整讲解：Lecture 6 §5.2「Clipped surrogate objective」。本节只补充：PPO 更新中的 advantage 如何估计，以及这个估计器为何需要在 bias 和 variance 之间折中。*

策略梯度更新需要一个对 $A^\pi(s_t,a_t)$ 的样本估计。完整 Monte Carlo return 方差较大，而单步 TD target 方差较低但依赖当前 value 估计，可能带来较大 bias。Lecture 7 用一条连续的 N-step 链把这两个端点连接起来。

课件回顾的另一个限制是：在 tabular 情况，policy space 是满足 $\pi_{sa}\ge0$ 且 $\sum_a\pi_{sa}=1$ 的行随机矩阵集合；policy-gradient 却在参数空间里走步，所以参数距离不等于 policy distribution 距离，步长很难直接按真实策略变化来选择。

这个问题和 Lecture 6 的 policy-space distance、PPO 动机相连，本讲后面的 KL bound 会把它写成可控制的变化代价。

PPO 的回顾公式只保留本讲后续要用的记号：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)},
\qquad
\mathcal L_{\theta_k}^{\mathrm{CLIP}}(\theta)
=\mathbb E\left[\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

这里 $\pi_{\theta_k}$ 是收集 rollout 的旧策略，$\pi_\theta$ 是当前正在优化的策略；本讲接下来重点解决如何得到 $\hat A_t$，而 clipping 的正负 advantage 分段行为仍以 Lecture 6 §5.2 的完整讲解为准。

本节约定 rollout transition 写作 $(s_t,a_t,r_t,s_{t+1})$，其中 $r_t$ 是从 $s_t$ 执行动作后得到的奖励；遇到 terminal state 时令后续 bootstrap value 为 $0$。这和前几讲使用的 $r_t$ convention 一致，但不要把它和有些教材的 $R_{t+1}$ 记号混读。

### 2.2 N 步优势估计（N-step advantage estimator）：从单步 TD 到完整 return

状态 value 估计器 $V$ 给出当前状态的 baseline。对 $k$ 步 lookahead，先收集 $k$ 个真实奖励，再用 $V(s_{t+k})$ 估计更远的尾部，得到

$$
\hat A_t^{(k)}
=
\sum_{l=0}^{k-1}\gamma^l r_{t+l}
+\gamma^k V(s_{t+k})-V(s_t).
$$

这个式子把“前 $k$ 步真实回报加上尾部 bootstrap”与当前状态 value 的差作为动作优势：结果为正，说明这次动作后的实际/估计回报高于 $V(s_t)$；结果为负则相反。$k=1$ 时它是一步 TD advantage，$k$ 越大越接近 Monte Carlo advantage。


定义 value [TD residual](academic-term-lookup:td%20residual)（也可理解为 value 的一步 prediction error）：

$$
\delta_t^V
=r_t+\gamma V(s_{t+1})-V(s_t).
$$

它比较“一步真实奖励加下一状态估计”与当前状态估计，输出是一个标量误差。

单个 TD error δt 就是一个Advantage， 就是
$$\boxed{ \hat A_t^{(1)} = r_t+\gamma V(s_{t+1})-V(s_t) = \delta_t }$$

但这里我们要考虑**N-step advantage**， 单步advantage 是就实际走了一步，后面就相信V(st+1)的预测。 N步的话，意思是可以实际往后在走N步，取得N个准确数据。2-step：
$$\hat A_t^{(2)} = r_t+\gamma r_{t+1} +\gamma^2 V(s_{t+2}) -V(s_t)$$
把它展开后你会发现他会消去，最终它等于：
$$\boxed{ \hat A_t^{(2)} = \delta_t+\gamma\delta_{t+1} }$$​​
所以：把相邻 residual 加起来会发生 [telescoping](academic-term-lookup:telescoping)：

$$
\begin{aligned}
\hat A_t^{(k)}
&=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}^V.
\end{aligned}
$$
展开右侧后，中间的 $V(s_{t+1}),\ldots,V(s_{t+k-1})$ 项逐项抵消，剩下的正是前一个 N-step 公式。因此，N-step estimator 既可以理解成一段带折扣的真实奖励，也可以理解成一串带折扣的 TD residual。

> [!example] 具体计算：两步 estimator 与 residual telescoping（说明用数据，非课件原例）
> 设 $\gamma=0.9$，$V(s_t)=2$，$V(s_{t+1})=3$，$V(s_{t+2})=0$，奖励为 $r_t=1$、$r_{t+1}=2$，且 $s_{t+2}$ 是 terminal state。
>
> $$
> \delta_t^V=1+0.9\times3-2=1.7,
> \qquad
> \delta_{t+1}^V=2+0.9\times0-3=-1.
> $$
>
> $$
> \hat A_t^{(2)}=1+0.9\times2+0.9^2\times0-2=0.8,
> $$
>
> 而 residual 形式给出 $1.7+0.9\times(-1)=0.8$。同一个数值说明了 telescoping 不是新的近似，而是同一估计器的重写。



### 2.3 GAE：用一个 $\lambda$ 平滑不同步长

广义优势估计（Generalized Advantage Estimation, GAE）不只选一个 $k$，而是把不同步长的 N-step estimator 做指数加权平均：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=(1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\hat A_t^{(k)}.
$$

这里 $\lambda\in[0,1]$ 控制“相信短期 bootstrap 还是延长真实回报”。将上式与 residual 的 telescoping 形式合并，可以得到更适合实现的表达：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V.
$$

在 residual 形式中，当前 residual 权重为 $1$，未来第 $l$ 个 residual 的权重为 $(\gamma\lambda)^l$；
因此 $\lambda$ 越小，估计器越快衰减、越依赖 value bootstrap，更依赖V，通常 variance 较低但 bias 较高。
$\lambda$ 越接近 $1$，后面的权重就越大，越想获取多一点的实际回报，估计器越接近长回报或 Monte Carlo，通常 bias 较低但 variance 较高。

这里的“较低/较高”是一般趋势，不是对每个环境和每个近似 value 的严格逐样本保证。

> [!example] 边界情况：$\lambda=0$ 与较大的 $\lambda$
> 当 $\lambda=0$ 时，只有 $l=0$ 项保留：
>
> $$
> \hat A_t^{\mathrm{GAE}(\gamma,0)}=\delta_t^V,
> $$
>
> 这就是 TD(0) advantage。若 $\lambda$ 接近 $1$，更远的 residual 仍有较大权重，估计通常更接近完整 return，但也会把更多 rollout 随机性带进 policy update。

课件的理解检查因此选择两项：$\mathrm{GAE}(\gamma,0)$ 使用 TD(0) advantage；它的 bias 通常比 $\lambda$ 接近 $1$ 的版本更大。课件把 $\lambda$ 接近 $1$ 的情况写成“更长的 return”，不要把它误说成“仍然是单步 TD”。

### 2.4 PPO 中的截断 GAE

实际 PPO 不会为了一个更新等待无限长 episode，而是在 rollout 窗口内截断：

$$
\hat A_t
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}^V.
$$
$$\boxed{ \hat A_t = \delta_t +\gamma\lambda\delta_{t+1} +(\gamma\lambda)^2\delta_{t+2} +\cdots }$$
这里 $T$ 表示本次 rollout 的结束位置

为什么T−t−1：

$\boxed{ t+l\le T-1 }$

因为最后一个可用 TD error 是：$\delta_{T-1}$​

所以：$l\le T-t-1$

因此求和才写：

$\boxed{ \sum_{l=0}^{T-t-1} }$​


比如真实 episode：s0→s1→s2→⋯→s10000​

PPO 可能只收：s0→s1→⋯→s2048，然后停下来，用这T=2048步去训练

假设 rollout 是：s0→s1→s2→s3→s4

有 4 个 TD error：δ0,δ1,δ2,δ3, 没有 δ4，因为这涉及到了计算s5

如果 t = 0，刚好就是：$\boxed{ \delta_0 +\gamma\lambda\delta_1 +(\gamma\lambda)^2\delta_2 +(\gamma\lambda)^3\delta_3 }$​​

​最后正好到δT−1​

现在从t = 2开始，计算advantage：

能使用的 TD error只有：$\delta_2,\delta_3$

所以：T−t−1=4−2−1=1

那么：$\hat A_2 = \sum_{l=0}^{1}(\gamma\lambda)^l\delta_{2+l}$​  ->. $\boxed{ \hat A_2 = \delta_2+\gamma\lambda\delta_3 }$​




对窗口末端的处理：

如果窗口在 terminal state 结束，后面就是没有状态了，最后的 bootstrap value 为 $0$；

如果只是时间截断，episode没有结束，是被人为停了，则照常需要用 value network (critic) 对窗口末端 bootstrap , 也就是V(sT) 来估计后面的未来。

于是 PPO 只需先运行环境 $T$ 个 timesteps，就能为窗口内每个动作计算一组 advantage，再进行多轮 minibatch 更新。


对应的数据流是：

`rollout -> V(s_t), V(s_{t+1}) -> δat -> reverse discounted sum -> advantage -> PPO clipped update`。

第一步：用旧策略 πol​d 跑环境，拿到：$(s_t,a_t,r_t,s_{t+1})$

第二步：critic 算 value：$V(s_t),\quad V(s_{t+1})$

第三步：每一步算 TD error：$\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$
从后往前计算时可以维护递推量

第四步：把 TD error 倒着累计：$A_t = \delta_t+\gamma\lambda A_{t+1}$

>	得到每个动作的 advantage：$A_0,A_1,\cdots,A_{T-1}$​

第五步：这些 At 交给 PPO

>	PPO 的核心目标大概就是：$r_t(\theta)A_t$
>	其中：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)} {\pi_{\text{old}}(a_t|s_t)}$


---

$$
\hat A_t=\delta_t^V+\gamma\lambda\hat A_{t+1},
$$
比如：

$\begin{aligned} \hat A_{2047} &=\delta_{2047}\\ \\ \hat A_{2046} &=\delta_{2046} +\gamma\lambda\delta_{2047}\\ \\ \hat A_{2045} &=\delta_{2045} +\gamma\lambda\delta_{2046} +(\gamma\lambda)^2\delta_{2047}\\ &\vdots \end{aligned}$

倒着进行计算


并在 episode 边界把下一项置零。这种递推与第 16 页的截断求和完全等价，却不需要为每个 $t$ 重复展开所有未来项。


## 3. 单调改进理论：为什么更新要保守

本节为lec6部分，策略改进ppo提高理论支撑

### 3.1 从 performance difference 到带 KL 代价的 bound

*首次完整讲解：Lecture 6 §4.2「Performance difference lemma」。本节只补充：将 state-distribution 近似升级为带 KL 惩罚的性能下界。*

Lec 6 只是说“如果新旧策略比较接近，那么 $d^{\pi'}\approx d^\pi$应该还行”；这里进一步问：到底多接近才算安全，lec6告诉你要怎么做：L−βKL。 现在解释这么做的理论依据，理论上为什么要限制策略变化的下界
$$\boxed{ \underbrace{J(\pi')-J(\pi)\ge L-C\,KL}_{理论解释} \quad \Longrightarrow \quad \underbrace{\max(L-\beta KL)}_{\text{实际算法的一种设计}} }$$​​​

Lecture 6 的 surrogate objective 用旧策略的 discounted state distribution $d^\pi$ 近似新策略的 $d^{\pi'}$。
Lecture 7 给出一个更保守的相对性能界。用 $C_\pi\ge0$ 表示理论中的常数，不等于上面的beta，那个是实际训练时的可调超参数，课件写成

$$
J(\pi')-J(\pi)
\ge
L_\pi(\pi')
-C_\pi\,\mathbb E_{s\sim d^\pi}
\left[D_{\mathrm{KL}}(\pi'\|\pi)[s]\right].
$$

右侧的第一项是旧策略数据可估计的 surrogate improvement，第二项是新旧 action distribution 的变化代价。

于是，候选策略只有在 surrogate 带来的收益足以支付 KL 变化时，理论下界才会为正；“小参数步”本身并不是这个保证的对象。

$$\boxed{ L_\pi(\pi') \text{ 是把真正的 }d^{\pi'} \text{ 换成旧策略 }d^\pi \text{ 后得到的 surrogate。} }$$
而$$\boxed{ -C_\pi KL }$$​

就是对这个替换造成的 **state-distribution mismatch 风险** 做一个理论上的安全扣分。

也就是说，**KL 并不是为了修正动作 probability ratio；动作已经被 importance ratio 修了。KL 真正防的是：策略改太远以后，连状态分布 dπ′ 都和旧数据的 dπ 不像了**

---

推导：

要证明的是：
即使我把 dπ′ 换成 dπ，这个误差也是有上界的。然后再证明这个上界由新旧策略的距离控制。


Performance Difference Lemma 给出：
$$\boxed{ J(\pi')-J(\pi) = \frac{1}{1-\gamma} \mathbb E_{ s\sim d^{\pi'}, a\sim\pi' } [ A^\pi(s,a) ] }$$​
Aπ(s,a)：这个动作相对于**旧策略 π** 有多好

我们假设新策略和旧策略比较接近，就可以对d进行近似，于是：
$$J(\pi')-J(\pi) \approx \frac{1}{1-\gamma} \mathbb E_{ s\sim d^\pi, a\sim\pi' } [A^\pi(s,a)]$$

右边这个东西，就定义成 surrogate improvement：
$$\boxed{ L_\pi(\pi') = \frac{1}{1-\gamma} \mathbb E_{ s\sim d^\pi, a\sim\pi' } [A^\pi(s,a)] }$$
然后再把动作分布用importance sampling 修一下。


我们定义：
$$\bar A(s) = \mathbb E_{a\sim\pi'}[A^\pi(s,a)]$$

那么真实提升是：
$$J(\pi')-J(\pi) = \frac{1}{1-\gamma} \sum_s d^{\pi'}(s)\bar A(s)$$

surrogate 是：(采用旧策略)
$$L_\pi(\pi') = \frac{1}{1-\gamma} \sum_s d^\pi(s)\bar A(s)$$

把真实提升和surrogate相减：
$$J(\pi')-J(\pi)-L_\pi(\pi') = \frac1{1-\gamma} \sum_s \left( d^{\pi'}(s)-d^\pi(s) \right) \bar A(s)$$

所以误差是由两个东西决定，新旧状态分布差异和Advantage的影响

>	我们发现，如果新旧两个策略相近，这两个东西都会变小


第一个“小”：策略接近 → 状态分布接近

>	整体而言：

$\boxed{ d^{\pi'}\approx d^\pi }$​

可以用 TV distance 正式衡量策略之间的距离：

$\alpha = \max_sD_{TV} \left( \pi'(\cdot|s),\pi(\cdot|s) \right)$

如果 α≪1

那么状态分布的偏移也会比较小。

粗略理解：

$\boxed{ d^{\pi'}-d^\pi = O(\alpha) }$


第二个“小”：策略接近 → $\bar A(s)$也小

>	因为：

$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$

而$V^\pi(s) = \mathbb E_{a\sim\pi}Q^\pi(s,a)$

所以：

$\mathbb E_{a\sim\pi}A^\pi = \mathbb E_{\pi}Q^\pi-V^\pi =0$

->
$$\boxed{ \mathbb E_{a\sim\pi} [A^\pi(s,a)] =0 }$$​这里不是单个advantage ， 而且是advantage 按照 动作a的概率 去做加权平均才会等于0


我们现在的$\bar A(s)$ 是：

$\bar A(s) = \sum_a\pi'(a|s)A^\pi(s,a)$

利用刚才那个 0：

$\sum_a\pi(a|s)A^\pi(s,a)=0$

所以：

$\bar A(s) = \sum_a\pi'(a|s)A^\pi(s,a) - \sum_a\pi(a|s)A^\pi(s,a)$

即：

$\boxed{ \bar A(s) = \sum_a [\pi'(a|s)-\pi(a|s)] A^\pi(s,a) }$​

证明完成：dπ′−dπ=O(α)


所以回到误差：

$\Delta = \frac1{1-\gamma} \sum_s \underbrace{ [d^{\pi'}(s)-d^\pi(s)] }_{O(\alpha)} \underbrace{ \bar A(s) }_{O(\alpha)}$​​

所以：$$\boxed{ \Delta=O(\alpha^2) }$$​
可得到：
$$\left| J(\pi')-J(\pi)-L_\pi(\pi') \right| \le C_\pi\alpha^2.$$
绝对值意味着：

$$-C_\pi\alpha^2 \le J(\pi')-J(\pi)-L_\pi(\pi') \le C_\pi\alpha^2$$

我们关心的是**最坏情况下，新策略到底能有多好**，所以取左边：
$$J(\pi')-J(\pi)-L_\pi(\pi') \ge -C_\pi\alpha^2$$


TV  变成  KL：

因为 TV distance 不太方便神经网络优化。

利用 Pinsker inequality：

$$D_{TV}^2(p,q) \le \frac12D_{KL}(p\|q)$$

也就是大致：

$$\alpha^2 \lesssim KL(\pi,\pi')$$

于是上面的：

$$J(\pi')-J(\pi) \ge L_\pi(\pi') - C_\pi\alpha^2$$

可以进一步写成类似：

$$\boxed{ J(\pi')-J(\pi) \ge L_\pi(\pi') - C_\pi KL(\pi,\pi') }$$
常数 1/2 等通常吸收到 Cπ​ 里。

$$\boxed{ J(\pi')-J(\pi) \ge L_\pi(\pi') - C_\pi KL }$$
告诉你：

真实提升≥旧数据估计出来的提升−近似误差

-> 用旧状态分布计算 Lπ(π′) 会有误差，但只要新旧策略 KL 很小，误差就被控制住


于是于是 TRPO出来了,他直接把KL卡死，给了KL一个固定的约束，在这个约束下去提高L。hard constraint

它一方面让

$$L_\pi(\pi')$$

尽可能大；

另一方面别让 KL 太大。

因此：

$$\boxed{ \max_{\pi'}L_\pi(\pi') }π′$$

同时限制：

$$\boxed{ KL(\pi,\pi')\le\delta. }$$​

得到：
$$\boxed{ \max_{\pi'}L_\pi(\pi') \qquad \text{s.t. } KL(\pi,\pi')\le\delta }$$


但hard constraint 对于神经网络的巨大参数，会比较麻烦。所以利用优化中的拉格朗日乘子。

把constraint写进目标函数，先写成：

$$KL(\pi,\pi')-\delta\le0$$

引入一个拉格朗日乘子：
$$\boxed{\beta\ge0}$$​

那么拉格朗日函数可以写成：$$\boxed{ L_\pi(\pi') - \beta \left( KL(\pi,\pi')-\delta \right) }$$​

展开：
$$= L_\pi(\pi') -\beta KL(\pi,\pi') +\beta\delta.$$

后者是常数，可以丢掉，所以就变成了lec6的公式，变成了PPO KL penalty。

理论下界→TRPO→PPO KL penalty→PPO clipping


[!important]
>	TRPO 的 KL constraint 有两种“简化实现思路”：

$$\boxed{ \begin{array}{ccc} & KL\le \delta &\\ \swarrow && \searrow\\ L-\beta KL && \text{PPO clipping}\\ \text{KL penalty版 PPO} && \text{clip版 PPO} \end{array}}$$​​

**KL penalty 和 clipping 是兄弟关系，不是前者严格推导出后者。**

---

举个实际 clip PPO rollout：

[!example]
>	现在有旧策略：πold

让它在环境中跑一条 trajectory：

τ:s0,a0,r0,s1,a1,r1,s2,a2,r2,⋯

例如采：T=1000个 timestep。

得到：(s0,a0),…,(s999,a999).

这些全部来自：πold

对于每个 timestep，计算：A^0,A^1,…,A^999

比如用你刚学过的 GAE。


然后对于每个样本：$r_t = \frac{ \pi_\theta(a_t|s_t) }{ \pi_{\rm old}(a_t|s_t) }$

再算：$\ell_t = \min \left( r_t\hat A_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t \right)$

最后：
$$\boxed{ L^{CLIP} \approx \frac1T \sum_{t=0}^{T-1}\ell_t }$$​

也就是说：
$$\boxed{ \mathbb E_t[\cdots] \approx \frac1T\sum_t[\cdots] }$$​
如果是N条轨迹 ， 那么就生成 N\*T 个transition

可以把它们全部摊平：

(s1,a1,A^1),…,(s1000,a1000,A^1000)

写成1/ N\*T



> [!example] 具体计算：surrogate 收益能否覆盖 KL 代价（说明用数据，非课件原例）
> 考虑只有一个状态的两动作问题。旧策略为 $\pi(a_1)=\pi(a_2)=0.5$，旧策略 advantage 为 $A^\pi(a_1)=1$、$A^\pi(a_2)=-1$；因此旧策略下 advantage 的期望为 $0$。
>
> 候选策略取 $\pi'(a_1)=0.6$、$\pi'(a_2)=0.4$，则
>
> $$
> L_\pi(\pi')
> =0.6\times1+0.4\times(-1)=0.2,
> $$
>
> $$
> D_{\mathrm{KL}}(\pi'\|\pi)
> =0.6\log(1.2)+0.4\log(0.8)\approx0.0201.
> $$
>
> 若这里只取说明用的 $C_\pi=2$，惩罚后的下界项约为 $0.2-2\times0.0201=0.1598>0$。这不是对真实环境性能的直接测量，而是展示 bound 如何把“收益”和“策略变化”放进同一个更新判据。


### 3.2 [Minorize--maximize](academic-term-lookup:minorize--maximize) 视角与改进保证

$$
J(\pi')-J(\pi)
\ge
L_\pi(\pi')
-C_\pi\,\mathbb E_{s\sim d^\pi}
\left[D_{\mathrm{KL}}(\pi'\|\pi)[s]\right].
$$

这节要我们理解下界(也就是右边这部分)为什么能变成 策略一定不会变差 的更新方法

理论构造的更新是

$$
\pi_{k+1}
=\arg\max_{\pi'}
\left[
L_{\pi_k}(\pi')
-C_{\pi_k}\mathbb E_{s\sim d^{\pi_k}}
\left[D_{\mathrm{KL}}(\pi'\|\pi_k)[s]\right]
\right].
$$

它把真实性能差的下界当成每轮要最大化的 surrogate，因此属于相对于真实目标的 Minorize--maximize 思路，构造下界然后最大化。证明只需检查旧策略是可行点：

$$
L_{\pi_k}(\pi_k)
=\mathbb E_{s,a\sim d^{\pi_k},\pi_k}[A^{\pi_k}(s,a)]=0,
\qquad
D_{\mathrm{KL}}(\pi_k\|\pi_k)[s]=0.
$$

所以整个下界：$$\boxed{ B_k(\pi_k)=0-0=0 }$$​
所以优化后的目标值至少为 $0$；在 bound 的假设和精确优化成立时，推出 $J(\pi_{k+1})-J(\pi_k)\ge0$。如果把策略限制在某个参数化类 $\Pi_\theta$，只要旧策略仍属于这个类，课件的证明形式仍然成立。
(因为已经有一个候选策略也就是旧策略，他的下界等于0，所以取最大化后，右边这个下界不可能比0小)

连起来：
$$\boxed{ J(\pi_{k+1})-J(\pi_k) \ge B_k(\pi_{k+1}) \ge0 }$$​

所以：
$$\boxed{ J(\pi_{k+1})\ge J(\pi_k) }$$​

这就是所谓的 **monotonic improvement（单调改进）**：
$$J(\pi_0) \le J(\pi_1) \le J(\pi_2) \le \cdots$$
理论上每次更新都不会退步。


由于 $L_{\pi_k}$ 和旧策略状态上的 KL 项都可以用 $\pi_k$ 采集的 samples 估计，这个更新不必为每个候选 policy 重新完整运行环境；这正是理论 bound 能转化为 data-efficient policy update 的关键。
也就是PPO可以**对同一批 on-policy 数据多轮训练**，不用每次sita参数一变，就立马重新跑数据，而是在同样的 st,at上重新计算：$\pi'(a_t|s_t)$，

这里的保证是有明确边界的，现实中PPO并没有严格保证每轮reward都上升：它依赖真实的期望、正确的 advantage、合适的常数和近似目标的精确最大化。
实际 PPO 用有限样本、近似 critic、多轮 SGD 和 clipping，因此应把“倾向于保守更新”与“每次真实回报必不下降”区分开。

比如PPO 实际用的是：$\hat A_t^{GAE}$​ ，它只是估计。
理论需要真正的期望：$\mathbb E_{s\sim d^{\pi_k}}[\cdots]$

实际只有有限样本：
$\sum_{t=1}^N(\cdots)$

理论假设你真的求出了：

$\arg\max_{\pi'}B_k(\pi')$

现实只是用 Adam 做几轮 SGD：

$L\theta \leftarrow \theta+\alpha\nabla_\theta$

根本不一定找到精确最大值。


### 3.3 从理论 bound 到 PPO

当折扣因子 $\gamma$ 接近 $1$ 时，理论常数可能很大，直接按 bound 选择的步长会过于保守。课件给出的两条实用路线是：调节 KL penalty ，或显式使用 trust-region KL constraint，也就是TRPO。PPO 的 adaptive KL penalty 和 clipped objective 正是这两种思想的易实现近似。

adaptive KL penalty 的更新可写成

$$
\theta_{k+1}
=\arg\max_\theta
\left[
\mathcal L_{\theta_k}(\theta)
-\eta_k\,\overline D_{\mathrm{KL}}(\theta\|\theta_k)
\right],
$$

其中 $\eta_k$ 会在迭代之间调节，使平均 KL 大致落在目标范围内；课件用 $\epsilon_k$ 表示这个 penalty coefficient，但本笔记用 $\eta_k$ 与 clipped objective 的区间参数 $\epsilon$ 区分。

*首次完整讲解：Lecture 6 §5.1「PPO 的两种近端更新思路」。本节只补充：单调改进理论解释了它们为什么要限制 policy-space 变化。*


#### PPO 课程总结

这部分可以归纳为四点：

1. **提高数据效率**：同一批 rollout 可以进行多次梯度更新。
2. **限制策略变化**：clipping 或 KL constraint 用于抑制过大的 policy shift。
3. **局部优化**：PPO 通常收敛到局部最优，而不能保证得到全局最优。
4. **兼容 actor--critic**：PPO 可以和 actor--critic 的 value method 结合。

课件还指出，policy-gradient 不要求 reward function 对 policy 参数可微，这正是 model-free policy optimization 的重要用途。课件将 PPO 概括为易实现、应用广泛的方法，并提到包括 ChatGPT tuning 在内的应用；这里属于课件中的课程定位，不把它扩展成未经核验的当前产品结论。


## 4. 从示范直接学习：Behavior Cloning 与 DAGGER

前面 PPO 一直在讲：

有 reward，agent 怎么靠 trial-and-error 学策略​

第四节开始讲 **Imitation Learning（模仿学习）**：

如果 reward 很难设计，但我们有专家示范怎么办？​

### 4.1 为什么需要 imitation learning

有些任务存在很好的决策策略，但由人手工设计 reward 往往困难。让人类在 agent 每一步之后提供 reward 可以避免手工写 reward，但交互次数和标注成本很高。

imitation learning 的替代思路是：让专家直接提供 [demonstration trajectories](academic-term-lookup:demonstration%20trajectories)，学习者从这些状态--动作序列中恢复想要的行为。

课件列出的应用包括复杂非结构地形导航（Silver et al., 2010）、模拟高速公路驾驶（Abbeel and Ng, 2004；Syed and Schapire, 2007；Majumdar et al., 2017）和停车场导航（Abbeel et al., 2008）。
共同点是：相比于手工设计一个能够准确刻画目标行为的 reward function，专家往往更容易直接通过示范告诉我们“应该怎么做”。因此，我们可以利用专家的行为数据来学习策略，而不要求专家直接给出一个完整的 policy。我们已知的专家策略$\pi^*$，我们并不知道他的具体函数，只能观察到他产生的行为数据，给他st，他会做动作at

**Imitation Learning 的核心设定** : 我们拥有由专家策略 $\pi^*$ 产生的一条或多条示范轨迹

$\tau_E=(s_0,a_0,s_1,a_1,\dots)$

但通常没有一个可直接用于强化学习的 reward function。目标是利用专家示范学习 learner policy πθ​。一般的 imitation learning 并不要求 transition model P(s′∣s,a)已知；某些经典 apprenticeship learning / IRL 理论会额外假设已知 MDP dynamics。

此时有三条不同问题：

- **Behavioral cloning**：直接用监督学习拟合专家 policy。
- **Inverse RL**：从示范反推出一个可能的 reward function $R$。
- **Apprenticeship learning via IRL**：用学到的 $R$ 再运行普通 RL，求出表现接近专家的 policy。

三者的输入相似，学习对象不同；尤其不要把“学会模仿动作”和“恢复专家真正优化的 reward”当成同一个目标。

我们已经有了专家policy，但它可能不适合最终部署，我们也不知道他的具体构成，它不是一个现成的神经网络，有可能它出决策的数据很慢，有可能是人类遥控操作。而接下来的操作类似把昂贵的专家能力变成便宜的可部署策略

### 4.2 行为克隆（behavioral cloning）：把示范变成监督数据

行为克隆（behavioral cloning, BC）固定一个 policy class，例如 neural network 或 decision tree，把每个示范时间步的 $(s_i,a_i)$ 当作监督样本，训练 policy $\pi_\theta$ 预测专家动作，最后可以做到给你状态，预测动作。假如给定了N个 已知的状态-动作对 ，离散动作常用负对数似然：

$$
\hat\theta
=\arg\min_\theta
\left[-\frac{1}{N}\sum_{i=1}^{N}\log\pi_\theta(a_i\mid s_i)\right].
$$

BC通常不是直接用专家策略，而是用它产生一批示范数据

它的作用很直接：在专家访问到的状态上，提高专家动作的概率；假设专家在状态si做了动作ai，那么模型就应该让这个πθ​(ai​∣si​) 尽可能大
它不需要知道 transition dynamics，也不需要先定义 reward。课件列出 ALVINN（Pomerleau, 1989）和飞行模拟器中的 learning-to-fly（Summut et al., 1992）作为早期成功案例，并提到 BCRNN 在机器人操作离线示范中的实践效果（Mandlekar et al., CoRL 2021）。

> [!example] 具体计算：BC 的一个小批量负对数似然（说明用数据，非课件原例）
> 三个示范状态上，模型给专家动作的概率分别为 $0.8$、$0.6$、$0.2$，则平均损失为
>
> $$
> -\frac{1}{3}[\log(0.8)+\log(0.6)+\log(0.2)]\approx0.781.
> $$
>
> 第三个状态的低概率贡献最大，所以一次梯度更新会更强地推动模型修正这个状态上的动作分布。因为第三个状态动作对就是专家给定的现实情况，但你模型居然给该状态下做这个动作的概率这么低，肯定要把它以高。这个计算只说明监督目标如何工作，不代表课件中的某一组实验数据。

BC 在数据覆盖充分、测试状态仍接近专家状态分布时可能表现很好；它的核心风险要到下一节才出现：监督学习的 iid 样本假设没有处理策略在时间上的反馈。


### 4.3 Compounding errors：训练和执行时访问的状态不同

监督学习通常假设训练与测试样本来自同一个分布，而且每个 $(s,a)$ 样本在时间上独立。
[MDP](academic-term-lookup:mdp) 中却有反馈回路：训练数据里的状态主要来自专家分布 $d^{\pi^*}$，部署时的状态来自学习策略自己的分布 $d^{\pi_\theta}$。

![[lec7-compounding-errors-p37.png|900]]

图示说明：学习策略一次偏离专家轨迹后，后续状态也可能偏离，因为模型之前学习的就是最标准的专家示范，但如果在一个地方偏离了，就会产生一个新情况，这个情况没有出现过，模型就无法准确预测，然后做出错误判断，又出现了一个新情况。这些状态在专家示范中没有被标注，于是模型没有“如何恢复”的训练数据。原图来自 `lecture/lecture7post.pdf` 物理 PDF 第 37 页。


若每一步在相关状态上的错误概率至多为 $\epsilon$，最初的独立误差直觉是总错误期望约为 $\epsilon T$。但一个早期错误可能让之后的每一步都进入错误分布，课件给出的近似链是

$$
\mathbb E[\text{total errors}]
\lesssim
\epsilon\bigl(T+(T-1)+(T-2)+\cdots+1\bigr)
=O(\epsilon T^2).
$$

这不是说每个任务的精确误差一定等于 $\epsilon T^2$；课件特别注明，正式结果需要 Ross et al. (2011) 的定理和假设。核心结论是：状态分布 mismatch 会把单步分类误差放大成随 horizon 更快增长的轨迹级误差。

这也是他和普通监督学习的不同，普通监督训练 测试集和训练集(x,y)都基本来自同个D，但BC，训练状态来自专家：

$$\boxed{ s_t\sim d^{\pi^*} }$$​

而实际部署以后，状态来自**你自己学出来的策略**：

$$\boxed{ s_t\sim d^{\pi_\theta} }$$


### 4.4 DAGGER：在学习策略自己的状态上请求专家标签

BC 的问题是：

> 我只在 **专家会访问的状态** 上问过专家。出了新状态，没有了专家数据，就不行了

那 DAgger 就说：

> **那我让 learner 自己去跑；它跑到哪里，我就在那些状态上问专家：“这里你会怎么做？”**


数据集聚合（DAGGER, Dataset Aggregation）修正的对象不是分类器结构(classifier structure)，而是训练数据的状态分布。

它修的是：

BC 数据：$s\sim d^{\pi^*}$，都来自于专家策略

DAgger 慢慢把训练数据扩展到：

$s\sim d^{\pi_1}, d^{\pi_2}, d^{\pi_3},\cdots$

它的循环包含五步：

1. 初始化空数据集和一个初始 policy。
2. 用当前 policy 与专家 policy 的混合策略走一段轨迹，使访问到的状态覆盖学习者真正可能遇到的区域。
3. 在这些 visited states 上查询专家动作标签。
4. 把新标签追加到旧数据集，重新训练 classifier。
5. 重复若干轮，并用 validation 选择最终 policy。

learner就是我们当前训练出来的策略pi

比如第一轮先拿专家数据：

$D_0=\{(s_1,a_1^*),(s_2,a_2^*),\ldots\}$  做 Behavior Cloning 时，

训练出一个神经网络：$\boxed{\hat\pi_1(a|s)}$​，

这个就是learner policy

expert是π∗ ， learner 是 π^1​,π^2​,π^3​....


完整流程：

最初：

$\pi^* \overset{\text{demonstration}}{\longrightarrow} D_0$​

然后：

$D_0 \overset{\text{BC}}{\longrightarrow} \boxed{\hat\pi_1}$​

注意：$\hat\pi_1$ 是 **learner**。

接着让 learner 在**环境**里 rollout，：

$\hat\pi_1 \longrightarrow s_1',s_2',s_3',\cdots$

然后对每一个 learner 遇到的状态询问 expert：

$s_1' \xrightarrow{\pi^*} a_1^*$

$s_2' \xrightarrow{\pi^*} a_2^*$​

把这些继续装入数据集，得到：$D_{\text{new}} = \{ (s_1',a_1^*), (s_2',a_2^*),\cdots \}$

合并：$\boxed{ D_1=D_0\cup D_{\text{new}} }$​​

再训练：

$D_1 \rightarrow \boxed{\hat\pi_2}$​

然后：

$\hat\pi_2 \rightarrow \text{环境 rollout} \rightarrow \text{新状态} \rightarrow \text{问 expert} \rightarrow D_2 \rightarrow \hat\pi_3$​

如此循环。


第 $i$ 轮的混合策略可写成

$$
\pi_i(a\mid s)
=\beta_i\pi^*(a\mid s)+(1-\beta_i)\hat\pi_i(a\mid s),
\qquad 0\le\beta_i\le1.
$$

当策略是 [deterministic](academic-term-lookup:deterministic) 时，这个 mixture 表示以概率 $\beta_i$ 让专家选择动作，以概率 $1-\beta_i$ 让当前 classifier 选择动作；它的作用是让 rollout 同时保留安全的 expert guidance 和 learner-induced states。

rollout(一次试验)时，beta概率让专家策略得出动作，1-beta概率让我们自己训练的learner得出动作。但是无论是让谁做出动作，到达一个状态，我们都会再问一遍专家，把正确的状态动作对存入数据集。

learner/mixed policy 跑出一批 visited states，然后这些 visited states 全部去问 expert 要标签。


数据流是：

`当前 policy -> 混合策略 rollout -> 学习者状态 -> expert action query -> aggregate dataset -> supervised retraining`。

![[lec7-dagger-p39.png|900]]

上图是课件第 39 页的源伪代码，原图来自 `lecture/lecture7post.pdf` 物理 PDF 第 39 页。理解它时应以行为顺序为主：$\beta_i$ 控制本轮有多少轨迹由专家参与，$\hat\pi_i$ 是上一轮学习到的 policy；新一轮 classifier 的训练数据包含所有历史标签，而不是只看本轮。



> [!example] 具体计算：DAGGER 的一次数据聚合迭代（说明用数据，非课件原例）
> 假设当前混合策略在三个状态 $s_0,s_1,s_2$ 上走出轨迹，专家查询返回动作 $a_0^*,a_1^*,a_2^*$。本轮新增数据是
>
> $$
> D_i=\{(s_0,a_0^*),(s_1,a_1^*),(s_2,a_2^*)\},
> \qquad
> D\leftarrow D\cup D_i.
> $$
>
> 下一轮 classifier 看到的是旧数据和这三条新数据的并集，因此即使 $s_1$ 是学习策略犯错后才访问到的状态，也会获得正确的恢复动作标签。

DAGGER 的收益是让最终 policy 在其**自身诱导的状态分布**下表现良好；课件把它概括为获得一个在该分布下表现良好的 stationary deterministic policy。代价是需要专家在学习策略访问的状态上反复标注，专家查询成本和混合策略的设计仍是关键限制。它不等于“只用一次示范就能解决所有分布错配”。



## 5. 逆强化学习（Inverse RL）：从示范反推 reward

普通强化学习: 

R(s,a) 已知⟹学习最优 policy π∗​

但现实中reward很难手写

所以出现逆强化学习：

专家 demonstrations ⟹ 推测专家使用的 reward​

BC想直接模仿 专家在这个状态做了什么动作

但IRL 希望学到的是靠近障碍物很不好，所以reward很低。这样子就算遇到没见过的新障碍，也可以根据reward 做规划。

### 5.1 特征奖励（feature-based reward）的设置与非唯一性

逆强化学习（inverse reinforcement learning, IRL）不直接拟合动作，而是假设专家策略近似最优，尝试找到一个 reward function，使专家行为在该 reward 下看起来合理。课件先给出一个重要边界：即使 transition model、示范和“专家最优”都已知，通常也存在无穷多个 $R$ 可以让同一专家策略最优，因此不能把示范当成足以唯一恢复真实 reward 的证据。

与其直接恢复每个状态到底值多少 reward，不如假设 reward 是若干“可解释特征”的加权和；然后只要 learner 和 expert 的这些特征长期出现频率接近，就能保证它们的 return 也接近。


### 5.2 线性特征奖励与 discounted feature expectations

课件限制 reward 为状态特征的线性组合：

$$
R(s)=w^\top x(s),
\qquad
w\in\mathbb R^n,
\quad
x:\mathcal S\to\mathbb R^n.
$$


![[Pasted image 20260807140100.png]]



其中 $x(s)$ 是已选定的 feature vector，$w$ 是需要从示范中学习的 reward 权重。把它代入从初始状态 $s_0$ 出发的 discounted value：

$$
\begin{aligned}
V^\pi(s_0)
&=\mathbb E_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t R(s_t)\mid s_0\right]\\
&=w^\top\mathbb E_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t x(s_t)\mid s_0\right]\\
&=w^\top\mu^\pi(s_0).
\end{aligned}
$$


这里 $\mu^\pi(s_0)$ 是从 $s_0$ 出发、按策略 $\pi$ 访问各状态 feature 的 [discounted expectation vector](academic-term-lookup:discounted%20expectation%20vector)。它把长轨迹压缩成一个向量，使价值比较变成 reward 权重与 feature frequency 的内积。


$$μ^π=E[x(s0​)+γx(s1​)+γ2x(s2​)+⋯]$$

叫：discounted feature expectation​

你可以把它理解成：

> **按照策略 π 行动以后，各种 feature 长期出现了多少。** 越远的未来权重越小，下面的是无折扣版


这样子:

policy决定 $μ^π$ ,也就是这些行为会产生哪些 feature

reward决定w，我到底有多喜欢这些 feature。

价值就是他们的乘积


若专家策略 $\pi^*$ 对真实权重 $w^*$ 最优，则

$$
(V^{\pi^*}(s_0)=V^*(s_0))\ge V^\pi(s_0),
\qquad \forall\pi,
$$
专家策略的reward带来的价值是最高的

等价地，在线性 feature 表示下有

$$
(w^*)^\top\mu^{\pi^*}(s_0)
\ge
(w^*)^\top\mu^{\pi}(s_0),
\qquad \forall\pi\ne\pi^*.
$$

这只是一个关于 $w^*$ 的不等式集合，通常不会唯一确定它。

专家行为无法唯一反推出 reward


### 5.3 特征匹配（feature matching）：不必恢复唯一 reward 也能保证性能接近


我们就不恢复 $w^∗$，直接让 learner 的 feature expectation 跟 expert 一样，即：$μ^π$≈$\mu^{\pi^*}$​

如果学习策略的 discounted feature expectation 接近专家：

$$
\left\|\mu^\pi(s_0)-\mu^{\pi^*}(s_0)\right\|_1\le\epsilon,
$$

并且 reward 权重满足 $\|w\|_\infty\le1$，每个权重绝对值最多是1，那么Holder 不等式给出

$$
\left|w^\top\mu^\pi(s_0)-w^\top\mu^{\pi^*}(s_0)\right|
\le\epsilon.
$$

所以 feature matching 是一种“足以保证所有 有界线性 reward 下表现 接近”的条件；它不是说恢复了专家真正的 $w$，也不是说 feature 相同的 stochastic policy 只有一个。

feature matching的意义就是：

如果行为 在 reward 所关心的所有 feature 上都跟 expert 差不多，那么无论这些 feature 如何加权(都小于1)，最终 reward 都不可能差很多

> [!example] 具体计算：$\ell_1$ feature matching 如何控制 reward gap（说明用数据，非课件原例）
> 令 $\mu^{\pi^*}=(3,1)$、$\mu^\pi=(2.95,1.03)$，则差向量的 $\ell_1$ 范数为 $0.05+0.03=0.08$。取 $w=(0.8,-0.5)$，有 $\|w\|_\infty=0.8\le1$。
>
> $$
> \left|w^\top\mu^\pi-w^\top\mu^{\pi^*}\right|
> =\left|0.8(-0.05)-0.5(0.03)\right|
> =0.055\le0.08.
> $$
>
> 计算展示的是 Holder bound 的数值含义：只要 feature 频率够接近，所有这类有界线性 reward 的差距都不会大于 feature mismatch 的上界。

Apprenticeship Learning / feature matching的 想法没BC那么严格，不用每个状态都模仿expert的动作，只要最终产生的关键行为特征差不多就行

例如两辆自动驾驶汽车：

Expert：

- 走左侧绕障碍；

Learner：

- 走右侧绕障碍。

动作序列完全不同。

但是可能：
$$μ ^{expert}≈μ ^{learner}$$
因为二者都：

- 没撞车；
- 保持安全距离；
- 速度合理；
- 顺利到达终点。

于是它们可能都有差不多的 return。


### 5.4 MaxEnt IRL 的目标：在满足示范统计量外不增加偏好

前一节仍有 ambiguity：同一 optimal policy 可对应无穷多 reward，同一 [feature count](academic-term-lookup:feature%20count) μ 也可能由无穷多个 stochastic policy 产生。 
满足同一组feature statistics 的行为可能有很多种，有很多不同的轨迹都满足，你应该选哪一条轨迹？feature statistics 也就是 一条 trajectory 上，各个 feature 累计出现了多少，就是μ 

5.3就只要求平均来看，learner 和 expert 产生类似的 feature就可以。Maximum Entropy发现满足：$μ^π=μ^{π^∗}$的 policy / trajectory distribution 可能有很多，MaxEnt 选择：

>	在所有 feature expectation 相同的 trajectory distributions 中，选 entropy 最大的。​


最大熵逆强化学习（maximum-entropy inverse reinforcement learning, MaxEnt IRL）用一个额外原则选择其中的路径分布。原则是：只满足示范 feature expectations(feature count 是对一条轨迹的统计，exception 是对很多条可能轨迹的平均统计)，不添加其他偏好。 所以他会避免出现在相同count的轨迹中，武断的选择其中一条。毕竟这几条都符合了你的条件

课件在同一方向页还列出 GAIL（Ho and Ermon, 2016），但本讲不展开其 adversarial training 细节。

对每条示范轨迹 $\tau_j$，课件这里使用**不折扣**的总 feature count：(上一节规定是有折扣的discount，所以会带权重γ)

$$
\mu_{\tau_j}=\sum_{s_i\in\tau_j}x(s_i),
\qquad
\tilde\mu=\frac{1}{m}\sum_{j=1}^{m}\mu_{\tau_j}.
$$

这和 §5.2 的 discounted policy-level $\mu^\pi(s_0)$ 不是同一个对象；前者是单条有限轨迹的统计量，后者是对整个策略policy 产生的轨迹分布和时间折扣取期望。

在 deterministic MDP 中，固定 horizon $H$ 后，policy 可以等价地看成所有可能 $H$-step paths 上的一个 distribution：

![[lec7-deterministic-path-distributions-p53.png|900]]

图中不同 path distribution 就是不同 policy 的另一种表示。原图来自 `lecture/lecture7post.pdf` 物理 PDF 第 53 页。

MaxEnt回答： 在平均 feature 一样的情况下，不同轨迹应该分别占多少概率？

它通过在满足 feature matching 的所有路径分布中，选择 entropy 最大的 来解决

最大熵原则写成约束优化：

$$
\begin{aligned}
\max_{P(\tau)}\quad
&-\sum_{\tau}P(\tau)\log P(\tau)\\
\text{s.t.}\quad
&\sum_{\tau}P(\tau)\mu_\tau=\tilde\mu,\\
&\sum_{\tau}P(\tau)=1.
\end{aligned}
$$


Entropy（熵）可以表示一个概率分布有多分散，多不确定，越不确定 它越大。利用P(τ)对 log 加权求和。 H(P)=−∑​P(τ)logP(τ)

第一行最大化 路径分布熵，后两行分别匹配 示范 feature statistics 和保证它是概率分布。

然后用拉格朗日乘子求这个最大化问题，求解出轨迹概率分布
在线性 reward 情况下，拉格朗日乘子就是 reward weight $w$，得到指数族路径分布：

$$
P(\tau\mid w)=\frac{1}{Z(w)}\exp\left(w^\top\mu_\tau\right),
\qquad
Z(w)=\sum_{\tau'}\exp\left(w^\top\mu_{\tau'}\right).
$$

这是一个 **softmax 分布**。


它告诉我们：

> **Maximum Entropy 条件下，最优的轨迹概率分布应该是什么形式。**

它按 $w^\top\mu_\tau$ 的高低指数 加权，完全相同分数的路径等概率；若把 reward 换成 cost，则相应权重的符号需要一起改变，不能把“高 reward”与“低 cost”混为同一 convention。

后续找一个 w (这就是第五节的目的¥)，让专家实际走出来的这些 trajectory 在 P(τ∣w) 下概率尽可能高


### 5.5 从最大熵路径分布学习 $w$

随机 MDP 中，一条 path 的概率还受到环境 dynamics 影响：

$$
P(\tau\mid w,P)
\propto
\exp\left(w^\top\mu_\tau\right)
\prod_{(s_i,a_i)\in\tau}P(s_{i+1}\mid s_i,a_i).
$$

因此同一个 reward weight 在随机环境下不只决定“偏好哪类 feature”，还必须和路径经过的 transition probability 一起计算。

$w^T\mu_\tau$ 表示 轨迹 τ 的总 reward ，对一条轨迹进行打分

如果环境是确定的：

> 执行动作 → 一定到达指定状态。

那么只考虑 reward 基本就可以。

但是现实中很多 MDP 是**随机的**。

比如机器人在状态 si​ 执行动作 ai​。

它不一定百分百到达某个状态，而是：$P(s_{i+1}\mid s_i,a_i)$
即便两条路径reward一样，但现实出现的概率依旧会不同

所以轨迹概率要加入 dynamics。

上面公式的第一部分表示: 这条轨迹有多好

第二部分表示: 这条路有多容易发生

假设一条轨迹是：

s1​→s2​→s3​→s4​

每一步成功概率分别为：
$0.9,\quad0.8,\quad0.95$

那么整条路径由于环境 dynamics 产生的概率就是：

0.9×0.8×0.95

所以一条轨迹的概率取决于两个东西


---

我们找w，就是认为：
>	如果某个 w 真的是专家的 reward，那么专家实际选择的这些轨迹，在这个 w 下面应该拥有很高的概率。


最大似然目标是

$$
w^*
=\arg\max_w L(w)
=\arg\max_w\sum_{j=1}^{m}\log P(\tau_j\mid w).
$$
找一个 w，让专家真正走过的那些轨迹出现的概率尽可能大。

L(w) 就是 likelihood（似然函数）的 log 形式 ， 它越大就代表此时的w把专家轨迹的概率设置的高，是合适的w


其梯度是 示范经验 feature count 与 模型期望 feature count 的差：

$$
\nabla L(w)
=\tilde\mu-\mathbb E_{\tau\sim P(\cdot\mid w)}[\mu_\tau]
=\tilde\mu-\sum_sD(s)x(s),
$$

其中 $D(s)$ 是当前 reward-induced path distribution (由当前 reward 所诱导出来的 轨迹概率分布) 下的 state visitation frequency(状态访问频率)。若示范更常到达某些 feature，而当前模型没有到达，梯度会提高相应 reward weight；反之则会降低它。

在某个时刻的访问某状态概率：
$D_t(s)=P(s_t=s)$

把所有时间步加起来：
$D(s)=\sum_tD_t(s)$

>	D(s) 实际上就是把**所有可能轨迹的概率**综合起来以后，计算某个状态平均会被访问多少次。

$$\sum_sD(s)x(s)$$
他是当前模型的 feature count

其中：

- μ~​：专家的 feature count
- x(s)：状态 s 有哪些 feature
- D(s)：**当前 reward 下，模型平均访问状态 s 多少次**

假设某个状态 s 有这样的 feature：
​​
$x(s)=\begin{bmatrix}1\\0\\1\end{bmatrix}$

意思是这个状态：

- 有 feature 1
- 没有 feature 2
- 有 feature 3

假设模型平均访问这个状态 5 次：

D(s)=5

那么这个状态贡献的 feature count 就是：
​​

$D(s)x(s)=5\begin{bmatrix}1\\0\\1\end{bmatrix}=\begin{bmatrix}5\\0\\5\end{bmatrix}$

然后把所有状态加起来：

$\sum_sD(s)x(s)$

就得到：

> 模型在当前 w 下产生的平均 feature count。



> [!example] 具体计算：MaxEnt IRL 梯度的方向（说明用数据，非课件原例）
> 假设只有两条可能路径，feature counts 为 $\mu_{\tau_1}=(1,0)$、$\mu_{\tau_2}=(0,1)$，示范各出现一次，所以 $\tilde\mu=(0.5,0.5)$。
>
> 当 $w=(\log3,0)$ 时，$P(\tau_1\mid w)=0.75$、$P(\tau_2\mid w)=0.25$，模型期望 feature count 为 $(0.75,0.25)$，因此
>
> $$
> \nabla L(w)=(0.5,0.5)-(0.75,0.25)=(-0.25,0.25).
> $$
>
> 梯度会降低第一个 feature 的权重、提高第二个 feature 的权重，使模型分布向示范的统计量靠近。


### 5.6 MaxEnt IRL 的前向--后向频次算法

**怎么高效算出这个 D(s)**？不能把世界上所有轨迹全部列出来再统计，因为轨迹数量可能指数爆炸

Backward 先算策略，Forward 再把状态访问概率一步一步往前传播，最后得到

当状态空间和 horizon 有限时，课件第 58 页用 backward pass 计算每个状态的 [partition value](academic-term-lookup:partition%20value)(Z(s): 从这里出发，所有未来路径的指数 reward 权重加起来有多少？)，再用 local action probability 和 forward pass 计算 visitation frequency。讲解顺序如下：

1. **Backward pass 初始化**：在终点设 partition value 为 $1$。
2. **向后递推 action partition**：对每个 $(s_i,a_i)$，把所有可能下一状态的 transition probability、当前 reward 的指数权重和下一状态 partition value 相乘后求和。
3. **状态 partition 与局部策略**：对动作求和得到 $Z(s_i)$，再令 $P(a_i\mid s_i)=Z(s_i,a_i)/Z(s_i)$。
4. **Forward pass**：从初始状态分布开始，按上述局部 action probability 和环境 transition 推进 $D_t(s)$。
5. **汇总频次**：对所有时间步的 $D_t(s)$ 求和，得到公式中的 $D(s)$，再用于 likelihood gradient。

用一般记号写，backward pass 的核心是

$$
Z(s,a)=\sum_{s'}P(s'\mid s,a)\exp\bigl(R_w(s)\bigr)Z(s'),
\qquad
Z(s)=\sum_a Z(s,a),
\qquad
\pi_w(a\mid s)=\frac{Z(s,a)}{Z(s)}.
$$

$\exp(R_w(s))$

表示：

> 当前状态 s 有多值得经过。 reward越高，这一项越大

Z(s′)

表示：

> 到了 s′ 之后，未来所有可能路径的总权重。


$P(s'|s,a)\exp(R_w(s))Z(s')$

就是：

> **当前 reward 权重 × 到达下一状态的概率 × 下一状态未来所有路径的权重。**

把所有可能到达的下一状态按照概率求和，得到这个动作有多值得选


然后Z(s)就是把这一状态所有动作都考虑进去，再利用它得到policy

$$\pi_w(a|s)=\frac{Z(s,a)}{Z(s)}$$

这实际上就是一个**归一化**。


所以Backward Pass 根据 reward 算出模型会怎么行动。


---

我们知道机器人在每个状态下选各种动作的概率。

于是现在可以问：

> **如果机器人真的按照这个 policy 从起点开始走，它会多经常经过每一个状态？**


forward pass 则从初始分布 $D_0$ 开始：

$$
D_{t+1}(s')
=\sum_{s,a}D_t(s)\pi_w(a\mid s)P(s'\mid s,a),
\qquad
D(s)=\sum_tD_t(s).
$$
$D_t(s)$

表示：

> 在第 t 个时间步，机器人处于状态 s 的概率。
> 

想知道下一时刻在 s′ 的概率是多少，就必须考虑所有可能从当前状态s到s'的方式


例如：
假设机器人一定从状态 A 出发：

$D_0​(A)$=1

其他状态：

$D_0(B)=D_0(C)=0$


最后拿计算出来的D(s) 乘上 x(s)，得到模型的feature count，然后专家 feature − 模型 feature 得到梯度，再更新w


课件第 58 页用 $Z_{s_i,0}=1$ 和有限 horizon 的索引写出同一递推；$R_w(s)$ 的指数位置取决于课程采用的 state-reward convention。

![[lec7-maxent-irl-algorithm-p58.png|900]]

这是课件第 58 页算法的源视觉；原图来自 `lecture/lecture7post.pdf` 物理 PDF 第 58 页。图像中的符号是有限 horizon 的递推版本：后向 pass 算归一化常数，前向 pass 算状态频次；它不是另一种 policy-gradient 更新。

该算法回答了第 57 页的问题：在已知 transition model 或能够与环境交互、获得 transition samples 时，可以计算模型期望的 $D(s)$；behavioral cloning 则只需要示范的 $(s,a)$ 对，不需要 transition model。



### 5.7 IRL 的边界与从 reward 回到 policy

MaxEnt IRL 最终学会 reward function，我们还要推出policy，这才是最终目的

$\text{Demonstration}\rightarrow\text{IRL}\rightarrow R_w\rightarrow\text{RL}\rightarrow\pi$

MaxEnt IRL 为多个可能 reward 提供了一个有原则的选择方式，但原始方法仍需要 transition model，或至少需要能在世界中模拟/行动以收集 transition information。
它和 BC 的数据需求不同：BC 可以直接在固定示范对上做 supervised learning。

得到 reward function 后，可以把它交给普通 RL 算法求 policy；也可以直接学习 desired policy。课件最后强调 imitation learning 能显著减少学到好策略所需的数据，但 reward ambiguity、专家标注成本、分布错配和 online RL 结合方式仍是开放问题。

课件还提醒，实际 reward learning 经常只有 preference pair，例如在两个 rollout $y_1,y_2$ 中选择更喜欢的一个；这种比较式反馈与 dueling bandits 相连，也是后续作业会继续出现的 setting。

BC：

直接学： s→a

也就是：

> 专家在这个状态做了什么，我直接模仿。

 IRL

则是：$(s,a)\text{ demonstrations}\rightarrow R\rightarrow\pi$

它试图理解：

> **专家为什么这么做？他背后的 reward 是什么？**

所以 IRL 理论上能够获得比单纯模仿更有泛化能力的目标描述。

但代价就是更复杂，而且传统 MaxEnt IRL 往往需要知道 transition dynamics：$P(s'|s,a)$




## 6. Importance Sampling 与 off-policy policy gradient

核心问题变成：

> **我想知道分布 P 下的期望，但我的数据偏偏来自另一个分布 Q，怎么办？**

### 6.1 重要性采样（importance sampling）：用另一分布估计期望

重要性采样（importance sampling）允许我们用从 proposal distribution $Q$ 采集的样本 估计目标 distribution $P$ 下的期望。只要满足 support condition：$P(x)>0$ 的地方也有 $Q(x)>0$，则

$$
\mathbb E_{x\sim P}[f(x)]
=\mathbb E_{x\sim Q}\left[\frac{P(x)}{Q(x)}f(x)\right]
\approx
\frac{1}{N}\sum_{i=1}^{N}
\frac{P(x_i)}{Q(x_i)}f(x_i),
\qquad x_i\sim Q.
$$

比值 $\frac{P(x)}{Q(x)}$ 就是 importance weight。估计器的方差可写成

$$
\operatorname{Var}(\hat\mu_Q)
=\frac{1}{N}\left(
\mathbb E_{x\sim P}\left[\frac{P(x)}{Q(x)}f(x)^2\right]
-\mathbb E_{x\sim P}[f(x)]^2
\right).
$$

但是当 $P/Q$ 在少数样本上很大时，单个加权样本会主导平均，方差可能爆炸；当 $Q$ 在 $P$ 有质量的区域为零时，估计式甚至失效。

>	一旦少数样本拿到巨大的权重，它们就会几乎决定整个估计结果，于是不同批次采样得到的结果会剧烈波动，这就是方差爆炸。

importance weight: 假设某类人在真正目标人口里： P(x)=0.5

但你的数据集里： Q(x)=0.1

那么：

$$\frac{P(x)}{Q(x)}=\frac{0.5}{0.1}=5$$


> 这种人在我的数据里严重采少了。

所以：

> **一个这样的样本，要当 5 个用。**


> [!example] 具体计算：一个极端 importance weight（说明用数据，非课件原例）
> 令 $P(x=1)=0.5$、$Q(x=1)=0.1$，并令 $f(1)=1$、$f(0)=0$。目标期望是 $0.5$，但从 $Q$ 采到 $x=1$ 时权重为 $0.5/0.1=5$，加权样本取值为 $5$；从 $Q$ 采到 $x=0$ 时取值为 $0$。
>
> $$
> \operatorname{Var}(\text{one weighted sample})
> =0.1\times5^2-0.5^2=2.25,
> $$
>
> 样本均值的方差是 $2.25/N$。它仍然可以无偏，但 proposal 与 target 越不匹配，有限样本越不稳定。


这张作为铺垫引出下一节的 Off-policy Policy Gradient。 off-policy 就是 我要优化 policy A，但是数据是 policy B 采的。


利用Importance Sampling，解决新旧 policy 不一样，但拿旧数据估计 新 policy 的 gradient




### 6.2 异策略策略梯度（off-policy policy gradient）：轨迹 ratio 会连乘

设目标策略是 $\pi_\omega$，但旧数据由行为策略 $\pi_{\omega'}$ 采集。 我们要怎么计算新策略的gradient。 Lecture 5/6 的 on-policy policy-gradient 形式可写成

$$
g
=\mathbb E_{\tau\sim\pi_\omega}
\left[
\sum_{t=0}^{T-1}
\gamma^t\nabla_\omega\log\pi_\omega(a_t\mid s_t)
A^\omega(s_t,a_t)
\right].
$$

τ ∼ $π_ω$​ 的意思不是“轨迹属于策略 πω​”；而是“轨迹 τ 是由策略 πω​ 生成（采样）出来的”。

轨迹 τ 是按照策略 πω​ 产生的，从 policy 诱导出的 trajectory distribution 采样。但在RL中，πω​

不是直接给 trajectory 概率。它只给：πω​(a∣s)，然后通过环境展开，

像上面的 x∼P ，表示从概率分布 P(x) 里面采样，直接给出概率



这里的 w 和reward 里的w不一样，前者是policy 网络的参数 输入s，输出π(a∣s)，通过改变参数，从而调整这个动作的概率。 后者是reward function 的权重参数

这里 $g$ 是对 policy parameters 的梯度向量；每个时间步的 score-function 向量由 [scalar advantage](academic-term-lookup:scalar%20advantage) 加权。

这里的 $A^\omega$ 是目标策略的 action advantage，不是行为策略自动提供的旧样本标签；实际算法还要用 critic、Monte Carlo 或其他估计器近似它。importance ratio 只改变轨迹的采样分布，不能凭空解决 advantage 估计问题。

>	用当前 policy 自己产生轨迹，然后根据 action 的 advantage 调整 policy

$\nabla_w\log\pi_w(a_t|s_t)$

score function梯度 决定：

> 参数应该往什么方向修改。


而： $A^w(s_t,a_t)$

决定：

> 这个 action 到底值得加强还是削弱。

但它要求 τ∼πw​

也就是： 数据必须由当前目标策略产生


把 prefix trajectory 记作 $\tau_{0:t}$，用 importance sampling 换到旧策略数据上时，每个时间步的 integrand 要乘

$$
\rho_{0:t}(\omega,\omega')
=\frac{P(\tau_{0:t}\mid\omega)}{P(\tau_{0:t}\mid\omega')}
=\prod_{i=0}^{t}
\frac{\pi_\omega(a_i\mid s_i)}{\pi_{\omega'}(a_i\mid s_i)}.
$$

初始状态分布 与 environment dynamics 在分子分母中抵消，留下策略 action probability 的连乘。关键风险是：即使每一步的策略差异都很小，许多接近 $1$ 的 ratio 乘在一起也会变得很大或很小。

$\tau=(s_0,a_0,s_1,a_1,\dots,s_T)$

它在 policy 下的概率包含每一步 action probability。

所以两个 policy 的轨迹概率比会变成：
$$\frac{P_{\pi_w}(\tau)}{P_{\pi_{w'}}(\tau)}=\prod_t\frac{\pi_w(a_t|s_t)}{\pi_{w'}(a_t|s_t)}$$

$P_w(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_w(a_t|s_t)P(s_{t+1}|s_t,a_t)$

$P_{w'}(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_{w'}(a_t|s_t)P(s_{t+1}|s_t,a_t)$

两个轨迹一除，再消去，就得到上面这个式子。
$$\frac{p(s_0)\prod_t\pi_w(a_t|s_t)P(s_{t+1}|s_t,a_t)}{p(s_0)\prod_t\pi_{w'}(a_t|s_t)P(s_{t+1}|s_t,a_t)}$$


所以最后把上面on-policy 的式子 转换成 off-policy gradient，做到拿旧数据估计 新 policy 的 gradient

$g=\mathbb{E}_{\tau\sim P_{w'}(\tau)}\left[\frac{P_w(\tau)}{P_{w'}(\tau)}\sum_t\gamma^t\nabla_w\log\pi_w(a_t|s_t)A^w(s_t,a_t)\right]$

$g=\mathbb{E}_{\tau\sim P_{w'}}\left[\left(\prod_{k=0}^{T-1}\frac{\pi_w(a_k|s_k)}{\pi_{w'}(a_k|s_k)}\right)\left(\sum_{t=0}^{T-1}\gamma^t\nabla_w\log\pi_w(a_t|s_t)A^w(s_t,a_t)\right)\right]$



> [!example] 边界情况：许多小 ratio 的连乘（说明用数据，非课件原例）
> 如果每一步的 ratio 都是 $1.1$，20 步 prefix 的权重为 $1.1^{20}\approx6.73$；如果每一步都是 $0.9$，则为 $0.9^{20}\approx0.122$。前者放大少数轨迹，后者让轨迹几乎没有贡献，这就是 exploding/vanishing importance weights。

因此，“用旧数据”与“保持估计稳定”之间存在直接冲突。PPO 通过限制单步 ratio 的有利变化来缓解这一点，但 clipping 是受控的 surrogate modification，不会把完整 prefix ratio 的方差问题凭空变成零；实际实现还需要注意 support、advantage 估计和旧 log-probability 的冻结。


### 6.3 本讲对 policy-gradient 的收束

本讲最后把 policy-gradient 的三层问题连在一起：GAE 改善 advantage 的 bias--variance；单调改进理论解释 KL/trust-region 约束的动机；importance sampling 说明为什么大量复用旧 policy 数据会有代价。

PPO 的吸引力在于它把这三者压缩成相对易实现的 conservative update，但它仍是非凸优化，通常只能保证找到局部最优附近的解。

## 7. Assignment Readiness

### 7.1 Assignment 2

- **前置知识已覆盖**：Lecture 6 的 PPO clipped objective、old-policy ratio、advantage、actor--critic 与 Assignment 2 的 `PPO.update_policy` shape 映射；本讲补充 GAE 和截断递推。
- **还未证明掌握**：尚未观察到你独立实现 reverse GAE、手算正负 advantage 的 clipping，或在 starter code 上跑通 sanity check。
- **结论**：现在可以开始 Assignment 2 的 PPO/advantage 部分，但“可以开始”不等于每个实现细节已经熟练。

### 7.2 Assignment 3

- **前置知识已覆盖**：本讲的 behavior cloning、compounding errors、DAGGER、feature-based reward、feature matching、MaxEnt IRL 和 preference-based reward learning 视角，覆盖 Assignment 3 中 supervised learning / reward learning / RLHF 背景所需的概念入口。
- **还需单独阅读**：Assignment 3 题面中的 Hopper reward engineering、Bradley--Terry preference model、reward-model cross-entropy、RLHF、DPO 公式和 starter-code 变量；这些不是本讲的完整教学对象。
- **mastery evidence**：目前没有 Assignment 3 的独立推导、代码运行或实验结果。
- **推荐下一步**：先阅读 Assignment 3 题面第 1--2 节，明确 reward engineering 与 preference-based reward learning，再将本讲的 BC/IRL 区分映射到题目要求。

## 8. 本讲必会公式

以下只列公式入口；完整语义、假设和数值计算见对应小节。

1. N-step advantage：

   $$
   \hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l r_{t+l}+\gamma^kV(s_{t+k})-V(s_t).
   $$

2. TD residual 与 GAE：

   $$
   \delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t),
   \qquad
   \hat A_t^{\mathrm{GAE}}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}^V.
   $$

3. Performance lower bound：

   $$
   J(\pi')-J(\pi)\ge L_\pi(\pi')-C_\pi\mathbb E_{s\sim d^\pi}[D_{\mathrm{KL}}(\pi'\|\pi)[s]].
   $$

4. Linear feature reward：

   $$
   R(s)=w^\top x(s),
   \qquad
   V^\pi(s_0)=w^\top\mu^\pi(s_0).
   $$

5. Feature matching guarantee：

   $$
   \|\mu^\pi-\mu^{\pi^*}\|_1\le\epsilon,
   \quad\|w\|_\infty\le1
   \Longrightarrow
   |w^\top\mu^\pi-w^\top\mu^{\pi^*}|\le\epsilon.
   $$

6. MaxEnt path distribution and gradient：

   $$
   P(\tau\mid w)=\frac{\exp(w^\top\mu_\tau)}{Z(w)},
   \qquad
   \nabla L(w)=\tilde\mu-\mathbb E_{\tau\sim P(\cdot\mid w)}[\mu_\tau].
   $$

7. Importance sampling：

   $$
   \mathbb E_P[f]=\mathbb E_Q\left[\frac{P}{Q}f\right]
   \approx\frac1N\sum_{i=1}^{N}\frac{P(x_i)}{Q(x_i)}f(x_i).
   $$

8. Off-policy prefix ratio：

   $$
   \rho_{0:t}=\prod_{i=0}^{t}\frac{\pi_\omega(a_i\mid s_i)}{\pi_{\omega'}(a_i\mid s_i)}.
   $$

## 9. 容易混淆点

- **GAE 的两个参数**：$\gamma$ 是 RL objective 的 discount factor；$\lambda$ 是 GAE 在不同 N-step estimator 之间的 bias--variance mixing coefficient。$\lambda=0$ 是 TD(0) residual，不是“没有 discount”。
- **N-step estimator 与 GAE**：$\hat A_t^{(k)}$ 选一个固定 $k$；GAE 对多个 $k$ 加权，PPO 常用的是有限 rollout 上的截断 residual sum。
- **terminal 与 time-limit truncation**：terminal 下一状态的 value 应为 $0$；仅因 rollout 窗口结束时仍可能需要 $V(s_T)$ bootstrap。
- **KL bound 与 PPO clipping**：理论 bound 的 $C_\pi$、adaptive-KL 的 penalty coefficient $\eta_k$ 都不是 PPO 的 clip hyperparameter $\epsilon$；前者是性能理论常数，中者调节平均 KL，后者限制单步 probability ratio。
- **单调改进的条件**：理论要求真实期望、正确的 advantage、合适的 bound 和精确优化；实际 PPO 的 clipping 不等价于每个 minibatch 后真实性能必不下降。
- **BC 与 IRL**：BC 拟合专家在已见状态上的 action；IRL 试图找到可解释 expert behavior 的 reward，通常不唯一。
- **BC 的数据需求**：BC 不需要 transition model；MaxEnt IRL 原始算法需要 transition model，或能模拟/行动以估计 visitation frequencies。
- **DAGGER 的混合策略**：$\beta_i$ 控制本轮专家参与程度，$\hat\pi_i$ 是学习策略；查询的是专家在学习策略访问状态上的动作，不是把专家轨迹简单复制一遍。
- **两种 feature count**：单条 MaxEnt 轨迹的 $\mu_\tau=\sum_i x(s_i)$ 按课件是不折扣的；policy value 的 $\mu^\pi$ 是带 $\gamma^t$ 的期望，不能直接互换。
- **feature matching 与 reward recovery**：匹配 feature expectations 可以给出有界线性 reward 的性能保证，但不证明真实 reward 或 policy 唯一。
- **importance sampling 的方向**：样本来自 $Q$，权重是 $P/Q$；不要把分子分母反过来。
- **单步 ratio 与 prefix ratio**：PPO 的 clipped objective 主要限制单步 $\pi_\omega/\pi_{\omega'}$；off-policy policy-gradient 的完整 prefix correction 是多个单步 ratio 的连乘。

## 10. 自测题

### 题目

1. 在固定 $V$ 下，写出 $\hat A_t^{(3)}$，并把它改写成三个 discounted TD residual。
2. 为什么 GAE 的 $\lambda=0$ 通常 bias 较高、variance 较低？$\lambda$ 接近 $1$ 时哪一项会增加？
3. 单调改进 bound 中，为什么旧策略 $\pi_k$ 能证明优化问题的目标值至少为 $0$？
4. BC 在训练数据上的 action accuracy 很高，为什么部署时仍可能失败？请用 $d^{\pi^*}$ 与 $d^{\pi_\theta}$ 解释。
5. DAGGER 的新数据与普通 BC 的原始数据相比，关键多了什么？
6. 为什么 feature matching 可以保证一类 reward 下的性能接近，却不能唯一恢复 $w$？
7. 写出 MaxEnt IRL 的 $P(\tau\mid w)$ 和 $\nabla L(w)$，解释梯度中 empirical count 与 model expected count 的方向。
8. 如果每一步 target/behavior policy ratio 都是 $0.95$，40 步 prefix ratio 大约是多少？这说明什么？
9. 课件的 REINFORCE understanding check 中，哪些说法成立：加入 baseline 可降低 variance；REINFORCE 保证全局最优；合适步长下可从次优确定性 policy 到局部最优；一步 policy-gradient 更新可能让 return 变差。

<details>
<summary>查看答案</summary>

1. $\hat A_t^{(3)}=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\gamma^3V(s_{t+3})-V(s_t)=\delta_t^V+\gamma\delta_{t+1}^V+\gamma^2\delta_{t+2}^V$。
2. $\lambda=0$ 只使用当前 TD residual，bootstrap 比重最大，所以通常 variance 小但 value 误差更直接进入 estimator；$\lambda$ 接近 $1$ 时使用更长的真实回报，通常 variance 增大、bias 减小。
3. $L_{\pi_k}(\pi_k)=\mathbb E_{d^{\pi_k},\pi_k}[A^{\pi_k}]=0$，且 $D_{\mathrm{KL}}(\pi_k\|\pi_k)=0$；旧策略是可行的零点。
4. 训练状态来自专家分布，部署状态由学习策略诱导；一次早期动作错误会把后续状态带到示范未覆盖区域，产生 compounding errors。
5. DAGGER 在学习策略自己访问到的 states 上请求 expert actions，并把这些标签聚合进数据集。
6. $|w^\top(\mu^\pi-\mu^{\pi^*})|\le\|w\|_\infty\|\mu^\pi-\mu^{\pi^*}\|_1$ 只给出有界权重下的性能差上界，不提供唯一性。
7. $P(\tau\mid w)=\exp(w^\top\mu_\tau)/Z(w)$；梯度是示范平均 feature count 减去当前模型期望 count，推动模型分布向示范统计量靠近。
8. $0.95^{40}\approx0.129$，权重明显衰减，说明 prefix importance sampling 可能出现 vanishing contribution。
9. 第一、第三、第四项成立；第二项不成立。policy-gradient 通常只谈局部优化，单步更新也可能因估计噪声或步长过大使 return 变差。

</details>

## 11. 本讲小结

GAE 用 $\lambda$ 在短期 TD 与长期 Monte Carlo 之间平衡，PPO 的截断实现可以用 reverse residual recursion 高效计算。单调改进理论把 surrogate gain 与 KL policy shift 放到同一个性能下界里，解释了 conservative policy update 的理论来源，但实际 clipping 仍不构成无条件的单调回报保证。

示范学习部分的分界也很清楚：BC 直接拟合 action，简单却受状态分布错配影响；DAGGER 通过在 learner-induced states 上查询专家来缓解 compounding errors。IRL 则进一步学习 reward，feature matching 能给出性能接近的 sufficient condition，MaxEnt IRL 用最大熵原则在 reward/path ambiguity 中选择一个分布，但需要 dynamics 或 transition samples。最后，off-policy policy-gradient 的 importance ratio 让旧数据可复用，却因 prefix ratio 连乘产生高方差，这也是 PPO 等方法强调近端更新的原因。

## 12. 延伸阅读

### 经典基础

- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016：GAE 的原始论文。
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017：PPO clipped objective 与 adaptive KL 视角。
- Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011：DAGGER 与 compounding-error 理论。
- Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning*, AAAI 2008；Abbeel and Ng, *Apprenticeship Learning via Inverse Reinforcement Learning*, ICML 2004：MaxEnt IRL 与 feature matching 基础。

### 前沿动态

截至 2026-08-03 核实：本讲不额外列独立的前沿动态条目。课件的 GAE、PPO、DAGGER、feature matching、MaxEnt IRL 和 off-policy importance sampling 已覆盖本讲需要的主线；加入未经当前材料支持的近期方法会降低 source fidelity。
