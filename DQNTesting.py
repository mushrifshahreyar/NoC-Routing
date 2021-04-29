import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import time
import random
import os
import pickle
from os import path

REPLAYMEM = "replay.dat"
VARIABLES = "variable.dat"
NACTIONS = 4
NROUTERS = 16
NSTATES = 2 * NROUTERS # Number of states - 32
GRIDSIZE = 4

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

LEARNINGRATE = 0.01
DISCOUNT = 0.9
EPSILON = 1
EPSILON_DECAY = 0.99955
MIN_EPSILON = 0.01

REPLAY_MEMORY_SIZE = 200  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 64  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 32  # Terminal states (end of episodes)


def oneHotEncode(my_id, dest_id):
    my_id_vec = [0] * NROUTERS
    dest_id_vec = [0] * NROUTERS
    my_id_vec[my_id] = 1
    dest_id_vec[dest_id] = 1
    my_id_vec.extend(dest_id_vec)
    return np.array(my_id_vec)

# Queries main network for Q values given current observation space (environment state)
def get_qs(model, my_id, dest_id):
#    model = tf.keras.models.load_model('./saved_model')
    state = oneHotEncode(my_id, dest_id)
    actions = model.predict(np.array(state).reshape(-1, NROUTERS*2))
    optimal_action = np.argmin(actions)
    return model, optimal_action

if __name__ == "__main__":
    print('Starting testing!')
    tm = tf.keras.models.load_model('./saved_target_model')

    while(1):
        while(1):
#            print("Waiting for file: Python")
            if(path.exists("input.txt")):
                break

        my_id = -1
        dest_id = 0

        while(True):
            #print("here")
            with open('input.txt','r') as f:
                lines = f.readlines()
                if(len(lines) == 2):
                    break

        my_id = int(lines[0])
        dest_id = int(lines[1])
      
        os.remove("input.txt")
        
        tm, action = get_qs(tm, my_id, dest_id)
        print('Action taken:', action)      
        my_x = my_id % GRIDSIZE
        my_y = my_id // GRIDSIZE
        while True:
            if(action == 0 and my_y < GRIDSIZE-1):
                 break
            elif(action == 1 and my_x < GRIDSIZE-1):
                 break
            elif(action == 2 and my_y>0):
                 break
            elif(action == 3 and my_x>0):
                 break
            elif(my_id == dest_id):
                 break
            else:
                 action = np.random.randint(0, NACTIONS);

        f = open("action.txt","w")
        f.write(str(action))
        f.close()
