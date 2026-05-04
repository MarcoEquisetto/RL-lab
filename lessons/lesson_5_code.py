import os, sys, numpy

# 1. Get the directory where this specific script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the path to the 'tools' folder relative to this script
module_path = os.path.abspath(os.path.join(current_dir, '..', 'tools'))

# 3. Add it to sys.path and import
if module_path not in sys.path: 
    sys.path.append(module_path)

from DangerousGridWorld import GridWorld

import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_rewards(cumulative_rewards_dyna_q, cumulative_rewards_dyna_q_plus):
    """
    Plots cumulative rewards over time steps.

    Args:
        cumulative_rewards_dyna_q: list of Dyna-Q rewards.
        cumulative_rewards_dyna_q_plus: list of Dyna-Q+ rewards.
    """

    time_steps_dyna_q = np.arange(len(cumulative_rewards_dyna_q))
    time_steps_dyna_q_plus = np.arange(len(cumulative_rewards_dyna_q_plus))

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps_dyna_q, cumulative_rewards_dyna_q, marker='o', linestyle='-', color='b', label='Dyna-Q')
    plt.plot(time_steps_dyna_q_plus, cumulative_rewards_dyna_q_plus, marker='x', linestyle='--', color='r', label='Dyna-Q+')
    plt.title('Cumulative Rewards Over Time Steps', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Cumulative Rewards', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ(environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99):
    """
    Implements the DynaQ algorithm
    """ 
    Q = numpy.zeros((environment.observation_space, environment.action_space))
    # Dictionary to act as the model. Maps state -> list of actions taken, 
    # and (state, action) -> (reward, next_state)
    M = {} 
    visited_states = []
    
    cumulative_rewards = []
    total_steps = 0

    # Run for a fixed number of episodes (using maxiters as episodes here)
    for episode in range(maxiters):
        # Assuming your environment has a reset() method to start a new episode
        environment.reset()
        S = environment.reset
        episode_reward = 0

        while not GridWorld.is_terminal(S):
            # 1. Choose action (Note the corrected argument order)
            A = epsilon_greedy(Q, S, eps)
            
            # 2. Take action
            R, S_prime = environment.step(A)
            episode_reward += R
            total_steps += 1
            
            # 3. Direct RL Update
            best_next_q = np.max(Q[S_prime])
            Q[S, A] += alfa * (R + (gamma * best_next_q) - Q[S, A])
            
            # 4. Model Update (Save experience)
            if S not in M:
                M[S] = {}
                visited_states.append(S)
            M[S][A] = (R, S_prime)
            
            # 5. Planning Phase
            for _ in range(n):
                # Sample a previously observed state
                sim_S = np.random.choice(visited_states)
                # Sample an action previously taken in that state
                sim_A = np.random.choice(list(M[sim_S].keys()))
                
                # Get simulated reward and next state from the model
                sim_R, sim_S_prime = M[sim_S][sim_A]
                
                # Simulated RL Update
                sim_best_next_q = np.max(Q[sim_S_prime])
                Q[sim_S, sim_A] += alfa * (sim_R + (gamma * sim_best_next_q) - Q[sim_S, sim_A])
            
            S = S_prime
            
            # Collect reward periodically (e.g., every 20 steps) to track progress
            if total_steps % 20 == 0:
                cumulative_rewards.append(episode_reward)

    policy = Q.argmax(axis=1) 
    return policy, cumulative_rewards


def dynaQplus(environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99, kappa=1e-3):
    """
    Implements the DynaQ+ algorithm
    """ 
    Q = numpy.zeros((environment.observation_space, environment.action_space))
    M = {}
    
    # Track time steps since a state-action pair was last visited
    time_since_visited = numpy.zeros((environment.observation_space, environment.action_space))
    visited_states = []
    cumulative_rewards = []
    total_steps = 0

    for episode in range(maxiters):
        environment.reset()
        S = environment.current_state()
        episode_reward = 0

        while not GridWorld.is_terminal(S):
            # 1. Epsilon-Greedy Action Selection
            A = epsilon_greedy(Q, S, eps)
            
            # 2. Take Action
            R, S_prime = environment.step(A)
            episode_reward += R
            total_steps += 1
            
            # --- DYNA-Q+ TIME TRACKING ---
            # Increment time for all transitions, then reset the one we just took
            time_since_visited += 1
            time_since_visited[S, A] = 0
            
            # 3. Direct RL Update
            best_next_q = np.max(Q[S_prime])
            Q[S, A] += alfa * (R + (gamma * best_next_q) - Q[S, A])
            
            # 4. Model Update
            if S not in M:
                M[S] = {}
                visited_states.append(S)
                # --- DYNA-Q+ UNTRIED ACTION MODELING ---
                # Assume untried actions lead back to the same state with 0 reward
                for a in range(environment.action_space):
                    M[S][a] = (0, S)
            
            # Overwrite with actual experience
            M[S][A] = (R, S_prime)
            
            # 5. Planning Phase
            for _ in range(n):
                sim_S = np.random.choice(visited_states)
                # Sample from all possible actions (since we initialized untried ones)
                sim_A = np.random.choice(list(M[sim_S].keys()))
                
                sim_R, sim_S_prime = M[sim_S][sim_A]
                
                # --- DYNA-Q+ EXPLORATION BONUS ---
                # Add bonus: R + kappa * sqrt(tau)
                sim_R += kappa * numpy.sqrt(time_since_visited[sim_S, sim_A])
                
                # Simulated RL Update
                sim_best_next_q = np.max(Q[sim_S_prime])
                Q[sim_S, sim_A] += alfa * (sim_R + (gamma * sim_best_next_q) - Q[sim_S, sim_A])
            
            S = S_prime
            
            if total_steps % 20 == 0:
                cumulative_rewards.append(episode_reward)

    policy = Q.argmax(axis=1) 
    return policy, cumulative_rewards

def main():
	print( "\n************************************************" )
	print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n5) Dyna-Q" )
	dq_policy_n00, _ = dynaQ( env, n=0  )
	dq_policy_n25, _ = dynaQ( env, n=25  )
	dq_policy_n50, dq_rewards = dynaQ( env, n=50  )
	env.render_policy( dq_policy_n50 )
	
	print( "\n5) Dyna-Q+" )
	dqp_policy_n00, _ = dynaQplus( env, n=0 )
	dqp_policy_n25, _ = dynaQplus( env, n=25 )
	dqp_policy_n50, dqp_rewards = dynaQplus( env, n=50 )
	env.render_policy( dqp_policy_n50 )
	print()
	
	print( f"\tExpected Dyna-Q reward with n=0:", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected Dyna-Q reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected Dyna-Q reward with n=50:", env.evaluate_policy(dq_policy_n50) )
	
	print()
	
	print( f"\tExpected Dyna-Q+ reward with n=0:", env.evaluate_policy(dqp_policy_n00) )
	print( f"\tExpected Dyna-Q+ reward with n=25:", env.evaluate_policy(dqp_policy_n25) )
	print( f"\tExpected Dyna-Q+ reward with n=50:", env.evaluate_policy(dqp_policy_n50) )
	
	plot_cumulative_rewards(dq_rewards, dqp_rewards)
	

if __name__ == "__main__":
	main()
