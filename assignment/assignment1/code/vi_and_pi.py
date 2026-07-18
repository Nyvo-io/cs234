### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

#python 通过你的实际赋值，从而确定变量类型
def bellman_backup(state, action, R, T, gamma, V): #当前在状态 state，执行动作 action，它的价值是多少
    """
    Perform a single Bellman backup.
    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states) -> T[当前状态, 动作, 下一个状态], 三维向量
    gamma: float
    V: np.array (num_states) -> v是一维向量

    Returns
    -------
    backup_val: float
    """

    backup_val = 0.
    backup_val = R[state, action] + gamma * np.sum(T[state, action, :] * V)


    #当前在状态 state，执行动作 action，它的价值是多少
    #T[1, 0, :]:当前在状态 1，执行动作 0，下一步到达所有状态的概率 -> [0.2, 0.5, 0.3]
    #T[state, action, :] * V : 行向量乘行向量
    #到状态0的概率 * 状态0的价值, 到状态1的概率 * 状态1的价值,


    return backup_val


####
#我不知道最终 V 是多少
#所以我先初始化 V = 0
#然后对每个 state 反复做 bellman_backup
#直到 V 稳定
####

def policy_evaluation(policy, R, T, gamma, tol=1e-3): #当前策略下的价值函数
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states) -> 选定的策略,在各个状态执行什么动作
    R: np.array (num_states, num_actions) -> 在这个状态选这个动作的奖励
    T: np.array (num_states, num_actions, num_states)ß -> 三维数组，概率
    gamma: float
    tol: float

        policy = [0, 1, 0, 2]

    意思是：
    在状态 0 选择动作 0
    在状态 1 选择动作 1
    在状态 2 选择动作 0
    在状态 3 选择动作 2
    policy[state]: 当前状态 state 下，策略规定应该选择哪个动作

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape #从R中取出信息，R是一维向量
    value_function = np.zeros(num_states) #value_function = [0., 0., 0., 0., 0.]

    #迭代求出value数组
    while True:
        new_value_function = np.zeros(num_states)
        for state in range(num_states): #求value数组每一位(状态)的值
            action = policy[state]
            new_value_function[state] = bellman_backup(state, action, R, T, gamma, value_function)
        
        diff = np.max(np.abs(new_value_function - value_function))

        value_function = new_value_function

        if diff < tol:
            break

    return value_function

    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int) #不加dtype，则默认是float

    #先算出某个状态下的所有动作的Q，然后挑选这个状态最大的那个Q值。 依此算出每一状态

    for s in range(num_states):
        Q_values = []
        for a in range(num_actions):
            Q = R[s,a]+ gamma * np.sum(T[s,a,:]*V_policy)
            Q_values.append(Q)

        new_policy[s] = np.argmax(Q_values)

    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.
    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

 
    while True:
        # 1. policy evaluation：计算当前 policy 对应的 V
        V_policy = policy_evaluation(policy, R, T, gamma, tol)

        # 2. policy improvement：根据当前 V_policy 找到更好的 policy
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        # 3. 如果新策略和旧策略一样，说明已经收敛
        if np.array_equal(new_policy, policy):
            break

        # 4. 否则继续用新策略进入下一轮
        policy = new_policy

    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.

    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """

    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    while True:
        new_value_function = np.zeros(num_states)

    #对每个状态 state：
    #    把所有 action 都试一遍
    #    算出每个动作的价值 Q
    #    取最大的 Q 作为这个状态新的 V

        for state in range(num_states):
            Q_values = np.zeros(num_actions)

            for action in range(num_actions):
                Q_values[action] = bellman_backup(
                    state, action, R, T, gamma, value_function
                )

            new_value_function[state] = np.max(Q_values)


        diff = np.max(np.abs(new_value_function - value_function))

        value_function = new_value_function

        if diff < tol:
            break

     #根据最终的 V_policy，提取最优 policy。
     #因为 value_function[state] 是“这个状态的最终价值”，但 policy 要的是动作编号
    for state in range(num_states):
        Q_values = np.zeros(num_actions)

        #对当前 state，把所有动作都试一遍，算每个动作的价值，选取最大的值(和value_function里的值相等)对应的动作
        for action in range(num_actions):
            Q_values[action] = bellman_backup(
                state, action, R, T, gamma, value_function
            )
        #提取策略，哪个 action 的 Q 值最大，这个状态就选择哪个 action
        policy[state] = np.argmax(Q_values)

    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'WEAK'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
