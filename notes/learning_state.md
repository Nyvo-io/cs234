# CS234 Learning State

Last updated: 2026-06-27

## Current Progress

- Completed: Lecture 1, `lecture/lec1/lecture1pre.pdf`
- Completed: Lecture 2, `lecture/lec2/lecture2pre.pdf`
- Notes written:
  - `notes/lec1_notes.md`
  - `notes/lec2_notes.md`
- Next lecture: Lecture 3, `lecture/lec3/lecture3pre.pdf`

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

## Assignment Readiness

- Assignment 1 Q1: ready to start; finite-horizon and infinite-horizon discounted reasoning have been covered.
- Assignment 1 Q2: ready to start; reward hacking and proxy reward issues have been covered conceptually.
- Assignment 1 Q3: ready to start; Bellman operators, contraction, fixed points, and greedy policies have been introduced.
- Assignment 1 Q4: ready to start; all required VI/PI coding formulas have been covered and mapped to starter-code variables.

现在可以开始写 Assignment 1。

## Active Follow-Up Checks

- When writing finite-horizon or infinite-horizon return formulas, verify whether the horizon symbol means total episode length or remaining horizon.
- Read and update `notes/confusions.md` when a notation ambiguity or recurring conceptual confusion appears.

## Recommended Next Step

Start Assignment 1, preferably Q4 RiverSwim coding first:

- implement `bellman_backup`
- implement `policy_evaluation`
- implement `policy_improvement`
- implement `policy_iteration`
- implement `value_iteration`
- run the provided sanity check for weak current and `gamma = 0.99`

If continuing lectures instead, start Lecture 3: model-free policy evaluation.
