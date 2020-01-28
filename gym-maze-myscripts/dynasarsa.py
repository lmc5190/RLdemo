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

                if(writeResults):
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
 
    '''
    Defining the simulation related constants
    '''
    max_episodes = 15
    render_maze = True
    optimal_steps = 62 #ONLY FOR 10X10 MAZE
    solution_streaks = 10 #number of streaks when maze is considered solved

    '''
    Creating a Q-Table for each state-action pair and environment model table
    '''
    q_table = np.zeros(n_states_tuple + (n_actions,), dtype=float)
    env_model = []

    #method parameters
    decaymultiplier_epsilon=32
    n=16
    decaymultiplier_alpha=2

    #defining outputfile
    writeResults = False
    outfile = None
    
    #run simulation for demo
    decay_factor_epsilon = decaymultiplier_epsilon* 10.0/np.prod(n_states_tuple, dtype=float)
    decay_factor_alpha =  decaymultiplier_alpha*10.0/np.prod(n_states_tuple, dtype=float)
    dynasarsa(planning_steps=n, run=1)

