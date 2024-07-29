# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:05:12 2023

@author: pedro
"""

# %% Import packages

import os
import copy
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import timeit

from torch import nn

from Case_Env import Pigeon
from RMA_Actor import Actor
from RMA_Critic import Critic
from RMA_Buffer_Replay import ReplayBuffer
from RMA_Buffer_Assimilation import AssimilationBuffer
from RMA_Noise import OUActionNoise, white_noise
from Step_4_Functions import digital_twin_update, get_ensemble
from Step_5_Functions import model_based_policy_update
from Step_6_Functions import select_optimal_policy


# %% RMA Algorithm


class RSA:

    def __init__(
            self,
            # --- Simulation setup --- #
            T_sim: float,  # Simulation time [s]
            dt_sim: float = 0.01,  # Simulation time-step [s]
            reward_type: str ="gaussian",
            windType='constant',
            windVelocity=0,
    ):

        # %%% Setup model-free parameters
        env = Pigeon(T_sim, dt_sim,reward_type=reward_type,windType=windType,
                     windVelocity=windVelocity)
        env.dt = 1 / env.simpleFlappingPropertiesEnv.properties['wing']['frequency'] / 100


        "Assign control problem parameters"
        self.env          = env  # Environment
        self.state_dim    = self.env.numObsState  # Size of the state vector
        self.state_bounds =  self.env.observation_space # ! td: Bounds of the state vector (as a gym.spaces.box)
        self.action_dim   = self.env.numActionState  # Size of the action vector
        self.max_action   = self.env.action_max #! td: float(self.env.action_space.high[0] )  # Bounds of the state vector (assumed symmetry in the action space [-1, 1])

        # Print case-specific information to the terminal
        print("state_dim={}".format(self.state_dim))
        print("action_dim={}".format(self.action_dim))
        print("max_action={}".format(self.max_action))

        # %%% Initialize logs
        self.train_log = {
            "actor_loss": [],
            "critic_loss": [],
            "episode": [],
            # 'grads': {'control': [], 'assimilation': []},
            "params": {"opt_control": [], "assimilation": [], "drl": []},
            "estimated_Q": [],
            "returns": [],
            "target_Q": [],
            "J_assimilation": [],
            "J_OC": [],
            "eval_mean_reward": [],
            "eval_std_reward": [],
            "noise": [],
        }

        self.environment_log = {
            "actions": [],
            "obs_states": [],
            "rewards": [],
            "d_k": [],
            "actions_eff": [],
        }
        self.episode_reward_list = []
        self.returns_list = []
        self.critic_loss_list = []
        self.reward_list =[]
        self.x_list =[]
        self.y_list =[]
        self.xd_list =[]
        self.yd_list =[]
        self.aphi =[]
        self.beta =[]
        self.Jp_list = []
        self.Ja_list = []
        self.Ja_mf_list = []
        self.weight_list = []
        self.policy_flag_list = []

        pass

    # %%
    def run_ep(
            self,
            max_episodes,
    ):
        # Initialize no. of time-steps
        total_timesteps = 0



        # Iterate over the number of episodes (n_e)
        for i_episode in range(1, max_episodes + 1):
            print("---------------------")
            print("Episode %i" % i_episode)
            print("---------------------")

            # Define the output path to store the episodes
            # path_out = self.log_folder + "/episodes/episode_" + str(i_episode) + "/"
            # os.makedirs(path_out, exist_ok=True)

            # Reset the environment at the start of the new episode (start at t=0)
            prev_state, d_k0 = self.env.reset(self.env.s0, True)

            # Reference/desired state (function of the disturbance)
            # s_tilde = self.env.get_reference(d_k0)

            # ??? Not sure why, but since the reference state is 0, it doesn' matter(?)
            # ??? In general, this sequence is a bit confusing to me (why extract prev_state and s_tilde if they're the same?)
            s = (prev_state - self.env.s_ref)/self.env.error_normalisation

            # Initialize reward as 0
            episode_reward = 0

            # Integrate over the the period T, with time-step dt
            done = False
            rew_list = []
            while not done:

                # Evaluate the policy for the current normalised error
                a = np.array([1]) #self.policy(s)

                # Perform the forward step of the model (using the correct dt)
                s_, r, done, log = self.env.forward_step(a)
                s_ = s_/self.env.error_normalisation

                # Logging
                self.reward_list.append(-1*r)
                self.y_list.append(s_[0]*5+5)
                self.yd_list.append(s_[1]*5)
                self.aphi.append((a[0]+1)/2*(88-50)+50)
                # self.beta.append((a[1]+1)/2*(30+30)-30)
                rew_list.append(-1*r)

                # Update the state
                s = s_

                # Previous state is read from the log-file
                prev_state = self.env.s

                # Update episode reward
                episode_reward -= r

                # Update number of time-steps
                total_timesteps += 1

                # Check if other ending conditions are met
                if done:
                    break



            self.episode_reward_list.append(episode_reward)
            print("Cumulative reward:", episode_reward)
            print("Final state:",self.yd_list[-1],self.y_list[-1])

#%%
# Dummy run

RSAenv = RSA(T_sim=1)

RSAenv.run_ep(1)

