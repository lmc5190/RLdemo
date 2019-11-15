import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define MC Learning function
def MCLearning(env, discount, alpha,  min_alpha,  epsilon, min_eps, episodes, episodes_stop_exploring):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = 9, high = 10, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    #compute alpha and epsilon decay
    decay_epsilon = (epsilon - min_eps)/episodes_stop_exploring
    decay_alpha = (alpha - min_alpha)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
    
        #reward and action stack for montecarlo update
        reward_stack = []
        action_stack = []

        #state stack for montecarlo update
        state_stack = []
        state_stack.append(state_adj)

        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 5):
                env.render()
                time.sleep(0.050)
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
            action_stack.append(action)
                
            # Get next state and reward
            state, reward, done, info = env.step(action)
            reward_stack.append(reward)

            # Discretize state, then add to stack
            state_adj = (state - env.observation_space.low)*np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            #Monte Carlo update on Terminal State
            #if done and state[0] >= 0.5:
            if done:
                G = 0
                while len(reward_stack) > 0:
                    G = reward_stack.pop() + discount*G
                    s1, s2 = state_stack.pop()
                    a = action_stack.pop() 
                    Q[s1,s2,a] = Q[s1,s2,a] + alpha*(G - Q[s1,s2,a])   
                
            # Adjust Q value for current state
            else:
               state_stack.append(state_adj) #append all nonterminal states
                                     
            # Update variables
            tot_reward += reward
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= decay_epsilon

        # Decay alpha
        if alpha > min_alpha:
            alpha -= decay_alpha

        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {} Alpha: {} Epsilon {}'.format(i+1, ave_reward, alpha, epsilon))
            
    env.close()
    
    return ave_reward_list

# Run MC Learning algorithm
rewards = MCLearning(env, 0.9, 0.3, 0.001, 1.0, 0.005,10000, 5000)
#def MCLearning(env, discount, alpha,  min_alpha, epsilon, min_eps, episodes, episodes_stop_exploring):

# Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('mc_rewards.png')
plt.show()     
plt.close()  