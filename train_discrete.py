import tensorflow as tf
import copy
import numpy as np
from discrete.agent_discrete import IAC
from discrete.network_discrete import CommNet
from mlagents.envs import UnityEnvironment
import matplotlib.pyplot as plt
import datetime
import time
import random
import os

# Hyperparameter save / load
save_path = "./save/"

# Number of Agents
state_size = 3
action_size = 4

num_agents = 3
num_predator = num_agents
num_prey = 1
num_relation = num_agents * (num_agents + 1)
num_relation_state = 2
num_external = 2

total_agents = num_predator + num_prey

EP_MAX = 500
EP_LEN = 5000
BATCH = 32

A_LR = 0.00001
C_LR = 0.0001

# Path of env
train_mode = True
env_name = "C:/Users/asdfw/Desktop/PP_env/New Predator Prey"
env = UnityEnvironment(file_name=env_name, worker_id = 1)
default_brain = env.brain_names[1]
brain = env.brains[default_brain]
env_info = env.reset(train_mode=train_mode, config={"Num_Agent": num_agents})[default_brain]

state = env_info.vector_observations
# State ====================
# [[ 1.   1.5  1. ]
#  [-3.  -2.5  0. ]
#  [-4.   1.5  1. ]
#  [-1.   1.5  1. ]]
# Move axis state ==========
# [[ 1.  -3.  -4.  -1. ]
#  [ 1.5 -2.5  1.5  1.5]
#  [ 1.   0.   1.   1. ]]
# Role ====================
# [1. 0. 1. 1.]
state = np.moveaxis(state, 0, -1)
role = state[2, :]
print("Role is {}".format(role))

prey_idx = []
for i in range(len(role)):
    if role[i] == 0:
        prey_idx.append(i)

net = IAC()

for ep in range(EP_MAX):
    ep_r = 0

    buffer_s, buffer_s_next, buffer_a, buffer_r, buffer_t = [], [], [], [], []
    temp_r = np.zeros([total_agents])

    for t in range(EP_LEN):

        # num_relation = 3 * 4 = 12
        # num_relation_state = 2
        # num_external = 2
        # total_agents = 4

        temp_action_prob = net.action_prob([state], False)
        action = []
        action_mat = np.zeros([action_size, total_agents])
        for i in range(total_agents):
            single_action = np.zeros([action_size])
            single_action[np.random.choice(np.arange(action_size), p=temp_action_prob[0][:, i])] = 1
            action.append(np.argmax(single_action))
            action_mat[:, i] = single_action[:]

        # Predator 와 Prey의 action을 나누어서 2번 수행하는 방식으로
        # 환경을 작동시킨다.
        # 서로 다른 role일때 action은 "4"로 넣어준다.
        action_prey = [4] * total_agents
        action_predator = [4] * total_agents
        for i, temp_action in enumerate(action):
            if i in prey_idx:
                action_prey[i] = temp_action
            else:
                action_predator[i] = temp_action

        # prey 의 action 진행
        done_step = 0
        env_info = env.step([action_prey])[default_brain]
        rewards1 = env_info.rewards
        terminals1 = env_info.local_done
        # prey 의 aciton 진행 후 게임이 끝난 경우
        # done_step = 1 set
        if True in terminals1:
            env_info = env.reset(train_mode=train_mode)[default_brain]
            done_step = 1
            ep_r += 1
        if train_mode is False:
            time.sleep(0.01)
        
        # predator 의 action 진행
        env_info = env.step([action_predator])[default_brain]
        rewards2 = env_info.rewards
        terminals2 = env_info.local_done

        # predator 의 action 진행 후 게임이 끝난 경우
        # done_step = 2 set
        if True in terminals2 and done_step == 0:
            env_info = env.reset(train_mode=train_mode)[default_brain]
            if done_step == 0:
                done_step = 2
                ep_r += 1

        # prey 의 action으로 게임이 끝난 경우
        # 게임의 termianl 과 reward는 prey action의 결과에서 나오는 환경 값을 사용
        # predator 의 action으로 인해서 게임이 끝난 경우 termianl은 or, reward는 두 reward를 더한다. 
        if done_step == 1:
            terminals = terminals1
            rewards = rewards1
        else:
            terminals = [a or b for a, b in zip(terminals1, terminals2)]
            rewards = [a + b for a, b in zip(rewards1, rewards2)]

        # Reward Shaping
        # prey의 reward가 -0.7보다 작은 경우
        # predator의 reward가 0.7보다 작은 경우에 0.3으로 reward를 조절한다.
        if rewards[np.argmin(role)] < -0.7:
            for i in range(len(rewards)):
                if role[i] == 1 and rewards[i] < 0.7:
                    rewards[i] = 0.3

        # Next state 진행 =========================
        state_next = env_info.vector_observations
        state_next = np.moveaxis(state_next, 0, -1)
        role = state_next[2, :]

        # Store transition ==================
        buffer_s.append([state])
        buffer_s_next.append([state_next])
        buffer_a.append([action_mat])
        buffer_r.append([[rewards]])
        buffer_t.append([terminals])
        # ===================================

        state = np.copy(state_next)

        # Train network ================================================================================================
        if (t + 1) % BATCH == 0 or t == EP_LEN - 1 and train_mode:
            bs, bs_, ba, br, bt = np.stack(buffer_s), np.stack(buffer_s_next), np.stack(buffer_a), np.stack(buffer_r), np.stack(buffer_t)
            loss = net.train_op(s=bs, s_next=bs_, a=ba, r=br, t=bt, c_lr=C_LR, a_lr=A_LR)
            buffer_s, buffer_s_next, buffer_a, buffer_r = [], [], [], []
            buffer_t = []
        # ==============================================================================================================
        if train_mode is False:
            time.sleep(0.1)
        temp_r += np.array(rewards)

    if (ep + 1) % 10 == 0:
        net.save_model(save_path)

    print('ep ', (ep + 1), ', count ', ep_r, ', rewards', temp_r, ', actor loss ', loss[0], ', critic loss ', loss[1])