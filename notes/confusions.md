# CS234 Confusions

这个文件记录容易反复混淆、需要后续 lecture 继续检查的点。新对话继续学习前应先读取它。

**使用方法**：
- 学习新 lecture 前，先浏览这个文件，提前知道哪些地方容易混淆。
- 做 assignment 时，如果遇到符号约定不清楚的地方，查阅对应条目。
- 发现新的易混淆点时，及时添加到这个文件中。
- 每个条目应包括：概念对比、标准约定、做题策略。

## Active Confusions

1. **Horizon $H$ 的含义**

   在公式里，$H$ 可能表示整个 episode 的总步数，也可能表示从当前时间 $t$ 开始剩余的步数。两种约定会导致 return 的求和上限不同。

   如果 $H$ 表示 episode 总步数，且 $t=0,\ldots,H-1$：

   $$
   G_t
   =
   \sum_{k=0}^{H-1-t}
   \gamma^k r_{t+k}
   $$

   如果 $H_{\text{rem}}$ 表示从当前时间 $t$ 开始剩余的步数：

   $$
   G_t^{(H_{\text{rem}})}
   =
   \sum_{k=0}^{H_{\text{rem}}-1}
   \gamma^k r_{t+k}
   $$

   **本课程默认约定**：除非题目或 lecture 明确说明，否则假设 $H$ 表示 episode 总步数，时间索引 $t=0,1,\ldots,H-1$。从时间 $t$ 开始的 return 应写成 $\sum_{k=0}^{H-1-t} \gamma^k r_{t+k}$。
   
   **做题时的策略**：做 Assignment 1 Q1 的 effective horizon 计算时，先仔细阅读题目对 $H$ 或 $H_{\text{rem}}$ 的定义，再选择对应的公式形式。

2. **Reward vs Value**

   奖励（reward）是一步反馈；价值（value）是未来折扣累计奖励的期望。

3. **Observation vs State**

   观测（observation）是智能体看到的东西；状态（state）是用于预测未来和决策的表示。当前观测不一定满足 Markov 假设。

4. **Evaluation vs Control**

   评估（evaluation）是判断给定策略有多好；控制（control）是寻找更好的策略。

5. **Large $\gamma$ vs short-term rewards**

   大的折扣因子 $\gamma$ 不是更看重短期。相反，$\gamma$ 越接近 1，未来奖励衰减越慢，长期奖励越重要。$\gamma=0$ 才是只看即时奖励。

6. **$B$ vs $B^\pi$**

   $B$ 是 Bellman optimality operator，会对动作取最大值：

   $$
   (BV)(s)
   =
   \max_a
   \left[
   R(s,a)
   +
   \gamma
   \sum_{s'}P(s' \mid s,a)V(s')
   \right]
   $$

   $B^\pi$ 是固定策略的 Bellman operator，不对动作取最大值：

   $$
   (B^\pi V)(s)
   =
   R(s,\pi(s))
   +
   \gamma
   \sum_{s'}P(s' \mid s,\pi(s))V(s')
   $$

   Assignment 1 Q3 同时使用这两个 operator，证明时不要混用。

7. **Arbitrary $V$ vs policy value $V^\pi$**

   Bellman residual 中的 $V$ 可以是任意价值向量，不一定对应任何真实策略。$V^\pi$ 则是某个策略 $\pi$ 在 MDP 中真实诱导出的价值函数。

8. **Finite-horizon policy is usually time-dependent**

   Infinite-horizon discounted MDP 中存在 stationary optimal policy；finite-horizon MDP 中，最优动作通常依赖剩余步数。同一个状态在剩 10 步和剩 1 步时可能应选择不同动作。
