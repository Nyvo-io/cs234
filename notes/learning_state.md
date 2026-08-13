# CS234 Learning State

Last updated: 2026-08-08

## Current Progress

- Coverage complete: Lecture 1, `lecture/lec1/lecture1pre.pdf`
- Coverage complete: Lecture 2, `lecture/lec2/lecture2pre.pdf`
- Coverage complete: Lecture 3, `lecture/lec3/lecture3pre.pdf`
- Coverage complete: Lecture 4, `lecture/lecture4post.pdf`
- Coverage complete: Lecture 5, `lecture/lecture5post.pdf`
- Coverage complete: Lecture 6, `lecture/lecture6post.pdf`
- Coverage complete: Lecture 7, `lecture/lecture7post.pdf`
- Coverage complete: Lecture 8, `lecture/lecture8post.pdf`
- Notes written:
  - `notes/lec_notes/lec1_notes.md`
  - `notes/lec_notes/lec2_notes.md`
  - `notes/lec_notes/lec3_notes.md`
  - `notes/lec_notes/lec4_notes.md`
  - `notes/lec_notes/lec5_notes.md`
  - `notes/lec_notes/lec6_notes.md`
  - `notes/lec_notes/lec7_notes.md`
  - `notes/lec_notes/lec8_notes.md`
- Concept registry: `notes/concept_index.md`
- Lecture 1–2 received source-fidelity and content-critical corrections during the v2 skill migration; their full presentation style remains compatible with legacy sections.
- Lecture 3 was fully restructured to the v2 note schema on 2026-07-19, with all 53 physical PDF pages covered, the Mars Rover source inconsistency recorded, and current LLM / embodied TD connections verified against primary sources.
- Lecture 4 was restructured against the complete post deck on 2026-07-22 and re-audited for first-teaching quality on 2026-07-24. Outside the learner-edited §3.3, each substantive first appearance now states its role and includes a minimal numeric calculation or complete trace; the note also records the source inconsistency in the page-26 Mars Rover MC action-value vector.
- Lecture 5 was strictly restructured against the complete 61-page post deck and completed the source/semantics, artifact-quality, and natural-language review passes on 2026-07-29. It embeds the Aliased Gridworld visual, carries one numerical example through the likelihood-ratio, REINFORCE, and baseline chain, and teaches REINFORCE narrative-first. Pre-only pages 62--78 are not counted as Lecture 5 classroom coverage; mastery evidence remains none recorded.
- Lecture 6 was strictly restructured against the complete 48-page post deck on 2026-07-28. It teaches VPG and actor-critic narrative-first, limits the score-norm baseline optimum to a fixed-state single gradient term, derives performance difference and surrogate/KL reasoning, explains both PPO variants, and includes three source-verified visuals plus Assignment 2 mappings. N-step is separately labeled as a source supplement from Lecture 5 pre pages 72--75.
- Lecture 7 was written against the complete 72-page post deck on 2026-08-03. It teaches N-step advantage and GAE with a shared numeric chain, gives the monotonic-improvement/KL bound and its feasibility proof, explains behavioral cloning, compounding errors, DAGGER, feature matching, MaxEnt IRL, and off-policy policy-gradient importance sampling narrative-first, and embeds four source-verified visuals. Mastery evidence remains none recorded.
- Lecture 8 was written against all 68 physical pages of `lecture/lecture8post.pdf` on 2026-08-04 and restructured on 2026-08-08. It distinguishes the PDF's physical pages from its internal 44-slide numbering, adds chapter relations and short roadmaps, teaches pairwise preference learning, Bradley--Terry reward modeling, RLHF's KL-regularized objective, DPO's closed-form policy transformation, and the Assignment 3 action-sequence/receding-horizon adaptation. It embeds four source-verified visuals and records the Lecture/Assignment label convention mismatch. Mastery evidence remains none recorded.
- Mastery evidence has not yet been recorded systematically; coverage completion does not imply mastery.
- Next lecture: Lecture 9, `lecture/lecture9post.pdf`

## Lecture 7 Covered

- N-step advantage estimators, value TD residuals, telescoping identity, and truncated GAE in PPO
- GAE bias--variance behavior as $\lambda$ moves from TD(0) toward long returns
- Monotonic improvement performance bound, majorize--maximize proof, KL/trust-region motivation, and practical PPO caveats
- Imitation learning motivation and the distinction among behavioral cloning, inverse RL, and apprenticeship learning
- Behavioral cloning objective, compounding errors, expert/learner state-distribution mismatch, and DAGGER dataset aggregation
- Linear feature reward, discounted feature expectations, expert optimality inequalities, feature matching guarantee, and reward/policy ambiguity
- Maximum-entropy IRL: trajectory feature counts, maximum-entropy constraints, exponential-family path distribution, stochastic dynamics, likelihood gradient, and forward--backward visitation computation
- Importance sampling expectation/variance and off-policy policy-gradient prefix ratios with exploding/vanishing weights

## Lecture 7 Assignment Readiness

- Assignment 2: GAE and truncated reverse recursion are covered; PPO ratio, clipping, advantage, and starter-code mappings were covered in Lecture 6. No independent implementation or sanity-check result has been observed.
- Assignment 3: BC, inverse RL, feature matching, Bradley--Terry preference modeling, reward-model cross-entropy, RLHF, and DPO are covered. Hopper reward engineering, environment-specific setup, code execution, and result interpretation still require separate work.
- Mastery evidence: none recorded for Lecture 7; coverage completion does not imply mastery.

## Lecture 8 Covered

- Human feedback as a spectrum from DAGGER/constant teaching to demonstrations only, with pairwise labels as a lower-burden comparison signal
- Bradley--Terry pairwise preference probability, reward-shift non-identifiability, Condorcet/Copeland/Borda winner definitions, and trajectory-level reward comparison
- Reward-model cross-entropy and the gradient with respect to trajectory reward difference
- RLHF pipeline: instruction tuning, preference collection, reward-model validation, and KL-regularized PPO/policy optimization
- DPO derivation from the KL-regularized closed-form optimal policy, partition-function cancellation, implicit reward, and final policy loss
- Assignment 3 mapping: reward-model labels, `RewardModel.update`, action-sequence distributions, SFT, frozen reference policy, DPO update, and receding-horizon control

## Lecture 8 Assignment Readiness

- Assignment 3 theory and implementation prerequisites: covered. The note explicitly maps the Bradley--Terry gradient, label conventions, RLHF objective, DPO loss, and starter-code shapes.
- Assignment 3 environment work: still required. Hopper reward terms, dataset-video judgments, three-seed PPO/RLHF/DPO runs, plots, and behavior comparisons have not been run or observed.
- Mastery evidence: none recorded for Lecture 8; coverage completion does not imply independent derivation, implementation, or experimental mastery.

## Lecture 1 Covered

- RL definition and motivation
- Optimization, delayed consequences, exploration, generalization
- Sequential decision process
- History, observation, state
- Markov assumption
- Problem types: bandits, POMDPs, Markov process, MRP, MDP
- Transition/dynamics model and reward model
- Deterministic and stochastic policies
- Evaluation vs control
- Markov process / Markov chain
- Markov reward process
- Horizon, return, value function
- Discount factor
- MRP Bellman equation
- Matrix Bellman equation and analytic solution
- Iterative dynamic programming for MRP value computation
- Model, policy, value function components of RL agents

## Lecture 2 Covered

- MDP definition
- MDP policy as a conditional action distribution
- MDP + policy induces an MRP
- MDP policy evaluation
- Deterministic policy space size
- MDP control and optimal policies
- Q-value / state-action value
- Policy improvement
- Policy iteration
- Bellman optimality operator
- Value iteration
- Bellman contraction
- Finite-horizon value iteration
- Simulation-based policy evaluation
- Mapping formulas to `assignment/assignment1/code/vi_and_pi.py`

## Lecture 3 Covered

- Model-free policy evaluation from direct experience
- Return/value notation and reward-indexing convention
- Bootstrapping
- First-visit and every-visit Monte Carlo evaluation
- Incremental Monte Carlo updates
- Bias, variance, MSE, and consistency
- Stochastic-approximation learning-rate conditions
- TD(0), TD target, and TD error
- Monte Carlo vs temporal-difference learning
- Certainty equivalence with maximum-likelihood MDP models
- Batch Monte Carlo vs batch TD convergence
- AB example and the role of the Markov assumption
- Connections to REINFORCE / PPO critics, TDRM for LLM process reward models, and TD-M(PC)$^2$ for embodied continuous-control planning

## Lecture 4 Covered

- Batch MC and batch TD policy evaluation, including the AB example and certainty equivalence
- Model-free policy iteration and the exploration-exploitation problem
- Epsilon-greedy policies and monotonic epsilon-greedy improvement
- Tabular Monte Carlo control and GLIE convergence
- On-policy versus off-policy learning
- Tabular Q-learning and SARSA, including terminal targets and the Mars Rover comparison
- Function approximation motivation, oracle regression, SGD, MC/TD value function approximation
- Semi-gradient TD and control with approximate Q functions
- Deadly triad: function approximation, bootstrapping, and off-policy learning
- DQN, experience replay, fixed Q-targets, pseudocode, and Atari ablations

## Lecture 5 Covered

- Value-based and policy-based RL, with actor-critic only as a route-map preview
- Stochastic policies and the Aliased Gridworld motivation
- Direct policy optimization, local optima, and policy gradients
- Trajectory distributions and likelihood-ratio / score-function gradients
- Dynamics-free trajectory-score decomposition
- Softmax and Gaussian policy score functions
- Policy gradient theorem
- Temporal causality, reward-to-go, and REINFORCE
- State baselines and the unbiasedness derivation
- Mapping REINFORCE and baseline formulas to Assignment 2 starter-code shapes

## Lecture 6 Covered

- Fixed-state single-term variance-optimal baseline and the $V^\pi(s)$ approximation
- Vanilla policy gradient training loop, value baselines, advantage, and actor-critic
- N-step target as an explicitly labeled source supplement from the Lecture 5 pre deck
- On-policy sample efficiency and limited reuse of old-policy data
- Parameter-space step size versus policy-space distribution change
- Normalized discounted state distribution $d^\pi$
- Performance difference lemma and its telescoping derivation
- Single-step policy probability ratios and remaining state-distribution mismatch
- Surrogate policy objective and KL approximation bound
- KL divergence between policies
- PPO with adaptive KL penalty
- PPO clipped objective for positive and negative advantages
- Mapping ratios, cached old log-probabilities, clipping, loss sign, and tensor shapes to `PPO.update_policy`

## Assignment Readiness

- Assignment 1 Q1: ready to start; finite-horizon and infinite-horizon discounted reasoning have been covered.
- Assignment 1 Q2: ready to start; reward hacking and proxy reward issues have been covered conceptually.
- Assignment 1 Q3: ready to start; Bellman operators, contraction, fixed points, and greedy policies have been introduced.
- Assignment 1 Q4: ready to start; all required VI/PI coding formulas have been covered and mapped to starter-code variables.

现在可以开始写 Assignment 1。

## Assignment 2 Readiness

- Ready to start: Assignment 2 §2.1 REINFORCE return computation, categorical/Gaussian policy distributions, and the mathematical basis of `PolicyGradient.get_returns` and `update_policy`.
- Ready to start: Assignment 2 §2.2--2.3 value baselines, MSE regression, sampled advantages, and advantage normalization.
- Ready to start: PPO clipping, old-policy probability ratios, cached rollout log-probabilities, and `PPO.update_policy` are covered in Lecture 6.
- Ready to start: Assignment 2 §3 trajectory/state distributions and the performance difference proof are covered across Lectures 5--6.
- Separate reading still required: the human-subjects ethics section relies on the Belmont Report, IRB, and compliance sources linked by the assignment rather than Lecture 6.
- Mastery evidence: none recorded for Lectures 5--6; no quiz, independent derivation, implementation, PPO tensor test, or Assignment 2 benchmark result has been observed yet.

现在可以开始 Assignment 2 的 policy-gradient、PPO 与 policy-induced distribution 部分；伦理部分需先完成题面指定阅读。

## Active Follow-Up Checks

- When writing finite-horizon or infinite-horizon return formulas, verify whether the horizon symbol means total episode length or remaining horizon.
- Verify whether a source writes the transition reward as $r_t$ or $R_{t+1}$ before computing returns.
- Keep first-visit MC, every-visit MC, TD targets, and full returns distinct.
- Keep behavior policy and target policy distinct when classifying on-policy and off-policy algorithms.
- Compare SARSA and Q-learning by the source of the next-action term before comparing their names or applications.
- Treat the Lecture 4 Mars Rover epsilon probability discrepancy as a source ambiguity; use the displayed epsilon-greedy formula.
- Treat the Lecture 4 Mars Rover MC $a_2$ vector as a source ambiguity; recompute returns from the displayed trajectory.
- Treat the Lecture 3 Mars Rover first-visit vector as a source ambiguity; align rewards to transitions and recompute from the displayed trajectory.
- When reading function approximation TD updates, distinguish a target's numerical value from the parameters through which gradients are allowed to flow.
- Keep the single-step policy probability $\pi_\theta(a\mid s)$ distinct from the complete trajectory probability $P(\tau;\theta)$.
- Keep sampled return $G_t$, expected action value $Q^\pi(s_t,a_t)$, and sampled advantage estimate $G_t-b(s_t)$ distinct.
- Check the discounted-objective convention before deciding whether an estimator includes an outer $\gamma^t$; Assignment 2 explicitly defines its required $G_t$.
- A standard policy-gradient baseline may depend on state but not on the action sampled at that time step.
- Remember that PyTorch optimizers minimize losses, while the mathematical policy objective is maximized.
- Keep parameter-space distance distinct from policy-space KL divergence.
- When using a policy ratio, identify which policy produced the actions, which policy is changing, and whether the state-distribution mismatch is also corrected.
- In PPO, keep rollout `old_logprobs` frozen throughout the minibatch updates; do not recompute the denominator with the current policy.
- Keep PPO log-probabilities, ratios, and advantages at shape `[batch]` to prevent accidental `[batch, batch]` broadcasting.
- Treat PPO clipping as a soft surrogate modification, not a hard KL constraint or a monotonic-return guarantee.
- Read and update `notes/confusions.md` when a notation ambiguity or recurring conceptual confusion appears.

## Recommended Next Step

If Assignment 1 is unfinished, start it, preferably Q4 RiverSwim coding first:

- implement `bellman_backup`
- implement `policy_evaluation`
- implement `policy_improvement`
- implement `policy_iteration`
- implement `value_iteration`
- run the provided sanity check for weak current and `gamma = 0.99`

If continuing lectures, start Lecture 7 with `lecture/lecture7post.pdf` and backreference Lecture 6 for policy performance, KL, and PPO foundations.

If starting Assignment 2, implement in this order:

- implement and test per-episode reverse return computation;
- verify categorical and diagonal-Gaussian log-probability shapes;
- implement the negative policy objective for optimizer minimization;
- test baseline prediction, advantage subtraction, and MSE shapes;
- implement `PPO.update_policy` with frozen old log-probabilities and `[batch]` ratio tensors;
- hand-check positive- and negative-advantage clipping before running environments;
- record sanity-check results as mastery evidence.
