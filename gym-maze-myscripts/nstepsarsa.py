import sys
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from collections import deque

import gym
import gym_maze

def simulate():

    # Instantiating the learning related parameters
    alpha = get_alpha(0)
    explore_rate = get_explore_rate(0)
    discount = 0.99
    num_streaks = 0

    # Render tha maze
    env.render()

    for episode in range(episodes):

        # Reset the environment, will not start at Terminal State
        state = env.reset()

        # the initial state and action
        state = state_to_bucket(state)
        action = select_action(state, explore_rate)
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

        while tau < T:
            tau=t-n+1
            print("t equals ", t)
            print("tau equals ", tau)
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
                if done:
                    T=t+1
                    print("terimnal t is ", T) 
                else:
                    action = select_action(state,explore_rate)
                    A.append(action)


            # Update Q, note how unneeded values are removed with popleft()
            #important note: n step algorithms use [n -1] Rewards and bootstrap for state and action value at timestep tau+n
            if(tau >= 0):
                print("length of S is ", len(S))
                print("length of A is ", len(A))
                s,a = S[0], A[0]

                #build appropriately size gamma vector, esp needed when episode terminates with t < n
                gamma_vector = [discount**(i+1) for i in range(len(R) - 1)]

                if(tau+n < T):
                    G = R.popleft() + np.dot(gamma_vector,R) + discount**(n-1)*q_table[S[len(S)-2] + (A[len(A)-2],)]
                elif(tau < T- 1):
                    G = R.popleft() + np.dot(gamma_vector,R)
                else:
                    G= R.popleft()
                q_table[s + (a,)] += alpha*(G - q_table[S.popleft() + (A.popleft(),)])
               
            #step t forward one step
            t=t+1

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Max Q: %f" % max_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % alpha)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % alpha)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if render_maze:
                env.render()

            if env.is_game_over():
                sys.exit()

            if t >= T+n-1:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= t_solved:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > terminal_streak:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        alpha = get_alpha(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(min_epsilon, min(0.8, 1.0 - math.log10((t+1)/decay_factor)))


def get_alpha(t):
    return max(min_alpha, min(0.8, 1.0 - math.log10((t+1)/decay_factor)))


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
    #env = gym.make("maze-sample-3x3-v0")
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
    min_epsilon = 0.001
    min_alpha = 0.2
    decay_factor = np.prod(n_states_tuple, dtype=float) / 10.0
    n = 10 #n-step sarsa

    '''
    Defining the simulation related constants
    '''
    episodes = 50000
    T = np.prod(n_states_tuple, dtype=int) * 100
    terminal_streak = 100
    t_solved = np.prod(n_states_tuple, dtype=int)
    DEBUG_MODE = 0
    render_maze = True
    enable_recording = False

    '''
    Creating a Q-Table for each state-action pair and environment model table
    '''
    q_table = np.zeros(n_states_tuple + (n_actions,), dtype=float)

    simulate()