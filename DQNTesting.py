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
GRIDSIZE = 8
NROUTERS = GRIDSIZE * GRIDSIZE
NSTATES = 2 * NROUTERS # Number of states - 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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
    actions = model.predict(np.array(state).reshape(-1, NSTATES))
    optimal_action = np.argmax(actions)
    return model, optimal_action

if __name__ == "__main__":
    print('Starting testing!')
    tm = tf.keras.models.load_model('./saved_target_model')
    legal_actions = 0
    total_actions = 0

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
        total_actions += 1
        print('Action taken:', action)      
        my_x = my_id % GRIDSIZE
        my_y = my_id // GRIDSIZE
        while True:
            if(action == 0 and my_y < GRIDSIZE-1):
                legal_actions += 1
                break
            elif(action == 1 and my_x < GRIDSIZE-1):
                legal_actions += 1
                break
            elif(action == 2 and my_y>0):
                legal_actions += 1
                break
            elif(action == 3 and my_x>0):
                legal_actions += 1
                break
            elif(my_id == dest_id):
                 break
            else:
                 action = np.random.randint(0, NACTIONS);
                 

        f = open("action.txt","w")
        f.write(str(action))
        f.close()
    print('Total actions: ', total_actions)
    print('Legal actions: ', legal_actions) 
