# CS234 Learning State

Last updated: 2026-07-04

## Current Progress

- Completed: Lecture 1, `lecture/lec1/lecture1pre.pdf`
- Completed: Lecture 2, `lecture/lec2/lecture2pre.pdf`
- Completed: Lecture 3, `lecture/lec3/lecture3pre.pdf`
- Notes written:
  - `notes/lec_notes/lec1_notes.md`
  - `notes/lec_notes/lec2_notes.md`
  - `notes/lec_notes/lec3_notes.md`
- Next lecture: Lecture 4, `lecture/lec4/lecture4pre.pdf`

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
- Connections to REINFORCE, PPO critics, robot learning, agent trajectories, and planning

## Assignment Readiness

- Assignment 1 Q1: ready to start; finite-horizon and infinite-horizon discounted reasoning have been covered.
- Assignment 1 Q2: ready to start; reward hacking and proxy reward issues have been covered conceptually.
- Assignment 1 Q3: ready to start; Bellman operators, contraction, fixed points, and greedy policies have been introduced.
- Assignment 1 Q4: ready to start; all required VI/PI coding formulas have been covered and mapped to starter-code variables.

现在可以开始写 Assignment 1。

## Assignment 2 Readiness

- Partially ready: Q2(a) return computation and the mathematical basis of `PolicyGradient.get_returns`.
- Partially ready: value baselines as MSE regression targets for variance reduction.
- Not ready as a whole: DQN/Q-learning, the policy gradient theorem, PPO clipping, and policy-induced distribution questions require later lectures.

现在还不应开始整份 Assignment 2。

## Active Follow-Up Checks

- When writing finite-horizon or infinite-horizon return formulas, verify whether the horizon symbol means total episode length or remaining horizon.
- Verify whether a source writes the transition reward as $r_t$ or $R_{t+1}$ before computing returns.
- Keep first-visit MC, every-visit MC, TD targets, and full returns distinct.
- Read and update `notes/confusions.md` when a notation ambiguity or recurring conceptual confusion appears.

## Recommended Next Step

If Assignment 1 is unfinished, start it, preferably Q4 RiverSwim coding first:

- implement `bellman_backup`
- implement `policy_evaluation`
- implement `policy_improvement`
- implement `policy_iteration`
- implement `value_iteration`
- run the provided sanity check for weak current and `gamma = 0.99`

If continuing lectures instead, start Lecture 4: model-free control.
