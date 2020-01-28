import sys
import numpy as np
import math
import random
import time
import pandas as pd
import csv
from collections import deque

import gym
import gym_maze

def dynasarsa(planning_steps, run=1):

    # Instantiating the learning related parameters
    alpha = get_alpha(0)
    epsilon = get_epsilon(0)
    gamma = get_gamma()
    num_streaks = 0
    solution_episode = -1

    # Render tha maze
    if render_maze:
        env.render()

    for episode in range(max_episodes):

        # Reset the environment
        state = env.reset()

        # the initial state
        state_0 = state_to_bucket(state)
        total_reward = 0

        max_timesteps=np.prod(n_states_tuple, dtype=int) * 100

        #metric tracking
        G_direct_vector = []
        G_indirect_vector = []
        dQ_direct_vector = []
        dQ_indirect_vector = []

        for t in range(max_timesteps):
            
            #time.sleep(0.02)

            # Select an action
            action = select_action(state_0, epsilon)

            # execute the action
            state, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(state)
            total_reward += reward

            # Update the Q and env_model based on the result
            q_sa = q_table[state + (select_action(state, epsilon),)]
            G_direct = reward + gamma * q_sa
            dQ_direct = G_direct - q_table[state_0 + (action,)]
            q_table[state_0 + (action,)] += alpha * dQ_direct
            env_model.append([state_0, action, reward, state])
            G_direct_vector.append(G_direct) #metric tracking
            dQ_direct_vector.append(dQ_direct)

            #Do Planning
            for n in range(planning_steps):
                s,a,r,sp = random.choice(env_model)
                q_sa = q_table[sp + (select_action(sp, epsilon),)]
                G_indirect = r + gamma * q_sa
                dQ_indirect = G_indirect - q_table[s + (a,)]
                q_table[s + (a,)] += alpha * dQ_indirect
                G_indirect_vector.append(G_indirect) #metric tracking
                dQ_indirect_vector.append(dQ_indirect)

            # Setting up for the next iteration
            state_0 = state

            # Render tha maze
            if render_maze:
                env.render()

            if env.is_game_over():
                break

            if done:
                T=t+1 #since agent is at t+1 while timeloop is at t
                if T == optimal_steps:
                    num_streaks = num_streaks + 1

                else:
                    num_streaks = 0

                if num_streaks == solution_streaks:
                    solution_episode = episode


                print("Episode %d finished after %d time steps with total reward = %f and on a %d game winning streak"
                      % (episode, T, total_reward, num_streaks))

                with open(outfile, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['dynasarsa', run, episode, T, np.mean(G_direct_vector), np.std(G_direct_vector), len(G_direct_vector), \
                                    np.mean(G_indirect_vector), np.std(G_indirect_vector), len(G_indirect_vector), \
                                    np.mean(dQ_direct_vector), np.std(dQ_direct_vector),\
                                    np.mean(dQ_indirect_vector), np.std(dQ_indirect_vector), alpha, epsilon, solution_episode, decay_multiplier])

                break

            elif t >= max_timesteps - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # Update parameters
        epsilon = get_epsilon(episode)
        alpha = get_alpha(episode)

def nstepsarsa(n, run=1):

    # Instantiating the learning related parameters
    alpha = get_alpha(0)
    epsilon = get_epsilon(0)
    gamma = get_gamma()
    num_streaks = 0
    solution_episode = -1

    # Render tha maze
    if render_maze:
        env.render()

    for episode in range(max_episodes):

        # Reset the environment, will not start at Terminal State
        state = env.reset()

        # the initial state and action
        state = state_to_bucket(state)
        action = select_action(state, epsilon)
        total_reward = 0

        #initialize deques
        S = deque()
        A = deque()
        R = deque()
        A.append(action)
        S.append(state)

        #intialize times
        T = float("inf")
        tau = float("-inf") #tau is timestep which for the updated action value's state and action
        t=0
        done = False

        #metric tracking
        G_vector=[]
        dQ_vector=[]

        while tau < T:
            tau=t-n+1
            #time.sleep(0.02)

            if t < T:
                # execute the action
                #important note: after this step, the agent is at time t+1 while the time loop is still at t
                state, reward, done, _ = env.step(action)

                # Observe the result
                state = state_to_bucket(state)
                total_reward += reward
                
                #Append reward (R_t) to deque
                R.append(reward)

                #append S_t+1 and select/append next action A_t+1
                S.append(state)
                #Record terminal time when you find it, otherwise select next action
                if done:  T=t+1
                else:
                    action = select_action(state,epsilon)
                    A.append(action)
                    #important note: the new action is only recorded, but not taken.


            # Update Q, note how unneeded values are removed with popleft()
            #important note: n step algorithms use [n -1] Rewards and bootstrap for state and action value at timestep tau+n
            if(tau >= 0):
                s,a = S[0], A[0]

                #build appropriately size gamma vector, esp needed when episode terminates with t < n
                gamma_vector = [gamma**(i+1) for i in range(len(R) - 1)]

                if(tau+n < T):
                    G = R.popleft() + np.dot(gamma_vector,R) + gamma**(n-1)*q_table[S[len(S)-2] + (A[len(A)-2],)]
                elif(tau < T- 1):
                    G = R.popleft() + np.dot(gamma_vector,R)
                else:
                    G= R.popleft()
                dQ= G - q_table[S.popleft() + (A.popleft(),)]
                q_table[s + (a,)] += alpha*dQ
                G_vector.append(G) #metric tracking
                dQ_vector.append(dQ)

            # Render tha maze
            if render_maze:
                env.render()

            if env.is_game_over():
                print("game over break")
                break

            #if t >= T+n-1:
            if tau == T-1:
                if T == optimal_steps:
                    num_streaks = num_streaks + 1
                else:
                    num_streaks = 0

                if num_streaks == solution_streaks:
                    solution_episode = episode

                print("Episode %d finished after %d time steps with total reward = %f and on a %d game winning streak"
                      % (episode, T, total_reward, num_streaks))

                
                with open(outfile, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['nstepsarsa', run, episode, T, np.mean(G_vector), np.std(G_vector), len(G_vector), \
                                    0,0,0, np.mean(dQ_vector), np.std(dQ_vector), 0, 0, alpha, epsilon, solution_episode, decay_multiplier])
                break
            elif t >= T - 1:
                pass

            #step t forward one step
            t=t+1

        # Update parameters
        epsilon = get_epsilon(episode)
        alpha = get_alpha(episode)
        print(solution_episode)


def select_action(state, epsilon):
    # Select a random action
    if random.random() < epsilon:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_epsilon(t):
    return max(min_epsilon, min(0.8, 1.0 - math.log10((t+1)*decay_factor_epsilon)))


def get_alpha(t):
    return max(min_alpha, min(0.8, 1.0 - math.log10((t+1)*decay_factor_alpha)))

def get_gamma():
    return 0.99

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_bounds[i][1]:
            bucket_index = n_states_tuple[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (n_states_tuple[i]-1)*state_bounds[i][0]/bound_width
            scaling = (n_states_tuple[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":

    # Initialize the "maze" environment, see __init__.py for more env names
    #env = gym.make("maze-random-5x5-v0")
    #env=gym.make("maze-sample-5x5-v0")
    env=gym.make("maze-sample-10x10-v0")
    #env=gym.make("maze-sample-100x100-v0")
    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    n_states_tuple = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

    # Number of discrete actions
    n_actions = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    min_epsilon = 0.0
    min_alpha = 0.0
    #decay_factor = 10.0/np.prod(n_states_tuple, dtype=float)
    #planning_steps = 10
    #n_nstepsarsa = 10

    '''
    Defining the simulation related constants
    '''
    max_episodes = 600
    render_maze = False
    optimal_steps = 62
    solution_streaks = 10 #number of streaks when maze is considered solved

    '''
    Creating a Q-Table for each state-action pair and environment model table
    '''
    q_table = np.zeros(n_states_tuple + (n_actions,), dtype=float)
    env_model = []

    #defining experiment values
    decay_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    #constants
    decaymultiplier_epsilon_dynasarsa=32
    decaymultiplier_epsilon_nstepsarsa=4
    n_dynasarsa=16
    n_nstepsarsa=4

    #defining outputfile
    outfile = '../experiments/dynasarsa_vs_nstepsarsa/alpha_decay/data/opt_epsdecay_n.csv'
    
    for decay_multiplier in decay_multipliers:
        decay_factor_epsilon = decaymultiplier_epsilon_dynasarsa * 10.0/np.prod(n_states_tuple, dtype=float)
        decay_factor_alpha =  decay_multiplier*10.0/np.prod(n_states_tuple, dtype=float)
        run=1
        for i in range(30):
            dynasarsa(planning_steps=n_dynasarsa, run=run)
            q_table = np.zeros(n_states_tuple + (n_actions,), dtype=float)
            env_model = []
            run=run+1

        decay_factor_alpha = decaymultiplier_epsilon_nstepsarsa * 10.0/np.prod(n_states_tuple, dtype=float)
        run=1
        for i in range(30):
            nstepsarsa(n=n_nstepsarsa, run=run)
            q_table = np.zeros(n_states_tuple + (n_actions,), dtype=float)
            run=run+1