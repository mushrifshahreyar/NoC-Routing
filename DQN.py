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
EPSILON = 0.01
EPSILON_DECAY = 0.9992
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


def initialize():
    # Main model
    model = create_model()

    # Target network
    target_model = create_model()
    target_model.set_weights(model.get_weights())

#    model.save('./saved_model')
#    target_model.save('./saved_target_model')
    # An array with last n steps for training
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

#    with open(REPLAYMEM, "wb") as f:
#        pickle.dump(replay_memory, f)

    # Used to count when to update target network with main network's weights
    target_update_counter = 0

#    with open(VARIABLES, "wb") as f:
#        pickle.dump(target_update_counter, f)
    return model, target_model, replay_memory, target_update_counter

def create_model():
    model = Sequential()

    model.add(tf.keras.Input(shape = (NSTATES, )))
    model.add(Dense(256, activation = 'relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(128, activation = 'relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(NACTIONS, activation='relu'))

    model.compile(loss="mse", optimizer=Adam(lr = LEARNINGRATE), metrics=['accuracy'])
    return model


# Adds step's data to a memory replay array
# (observation space, action, reward, new observation space, done)
def update_replay_memory(replay_memory, my_id, dest_id, prev_router_id, prev_action, queueing_delay, done):
#    replay_memory = []
#    with open(REPLAYMEM, "rb") as f:
#        replay_memory = pickle.load(f)

    transition = [my_id, dest_id, prev_router_id, prev_action, queueing_delay, done]
    replay_memory.append(transition)

#    with open(REPLAYMEM, "wb") as f:
#        pickle.dump(replay_memory, f)
    return replay_memory


# Trains main network every step during episode
def train(model, target_model, replay_memory, target_update_counter, my_id, dest_id):

#    replay_memory = []
#    with open(REPLAYMEM, "rb") as f:
#        replay_memory = pickle.load(f)

    # Start training only if certain number of samples is already saved
    if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return model, target_model, replay_memory, target_update_counter

#    model = tf.keras.models.load_model('./saved_model')
#    target_model = tf.keras.models.load_model('./saved_target_model')
    # Get a minibatch of random samples from memory replay table
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

    # Get current states from minibatch, then query NN model for Q values
    current_states = np.array([oneHotEncode(transition[2], transition[1]) for transition in minibatch])
    current_qs_list = model.predict(current_states)

    # Get future states from minibatch, then query NN model for Q values
    # When using target network, query it, otherwise main network should be queried
    new_current_states = np.array([oneHotEncode(transition[0], transition[1]) for transition in minibatch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    y = []

    # Now we need to enumerate our batches
    for index, (my_id, dest_id, prev_router_id, prev_action, reward, done) in enumerate(minibatch):

        # If not a terminal state, get new q from future states, otherwise set it to 0
        # almost like with Q Learning, but we use just part of equation here
        if not done:
            min_future_q = np.min(future_qs_list[index])
            new_q = reward + DISCOUNT * min_future_q
        else:
            new_q = reward

        # Update Q value for given state
        current_qs = current_qs_list[index]
        current_qs[prev_action] = new_q

        # And append to our training data
        X.append(oneHotEncode(prev_router_id, dest_id))
        y.append(current_qs)

    # Fit on all samples as one batch, log only on terminal state
    model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)
    #print('TRAINING INSIDE train() FUNCTION')
    # save and load target_update_counter
    # Update target network counter every episode
#    target_update_counter = 0
#    with open(VARIABLES, "rb") as f:
#        target_update_counter = pickle.load(f)

    if done:
        target_update_counter += 1
#        with open(VARIABLES, "wb") as f:
#            pickle.dump(target_update_counter, f)

    # If counter reaches set value, update target network with weights of main network
    if target_update_counter > UPDATE_TARGET_EVERY:
        target_model.set_weights(model.get_weights())
        target_update_counter = 0
#        target_model.save('./saved_target_model')
    
#    model.save('./saved_model')
    return model, target_model, replay_memory, target_update_counter



# Queries main network for Q values given current observation space (environment state)
def get_qs(model, my_id, dest_id):
#    model = tf.keras.models.load_model('./saved_model')
    state = oneHotEncode(my_id, dest_id)
    actions = model.predict(np.array(state).reshape(-1, NROUTERS*2))
    optimal_action = np.argmin(actions)
    return model, optimal_action


print("\n\nSTARTING PYTHON")
#print("TF VERSION", tf.__version__)

#print("READING VALUES")
#isInit = int(input())
#my_id = int(input())
#dest_id = int(input())

#prev_router_id = int(input())
#prev_action = int(input())
#queueing_delay = int(input())
#print("READING DONE")


#if(isInit == 0):
#    initialize()

#action = get_qs(my_id, dest_id)
#print('ACTION FROM PYTHON', action)
#f = open("action.txt", "w")
#f.write(str(action))
#f.close()

#done = 0
#if(my_id == dest_id):
#    done = 1

#update_replay_memory(my_id, dest_id, prev_router_id, prev_action, queueing_delay, done)

#print("TRAINING STARTED")
#train(my_id, dest_id)

#print("PYTHON EXECUTED")

if __name__ == "__main__":
#    print("Started")
    iterations = 0
    EPISODES = 700
    destcount = 0    
    m = None
    tm = None
    rm = None
    tuc = None

    if EPISODES == 0:
        m, tm, rm, tuc = initialize()
    else:
        m = tf.keras.models.load_model('./saved_model')
        tm = tf.keras.models.load_model('./saved_target_model')
        with open('REPLAYMEM', 'rb') as f:
            rm = pickle.load(f)
        with open('VARIABLES', 'rb') as f:
            tuc = pickle.load(f)


    while(1):
        iterations += 1
        EPISODES += 1
        print('Episode:', EPISODES)
        print('Iteration:', iterations)
        while(1):
#            print("Waiting for file: Python")
            if(path.exists("input.txt")):
                break

        my_id = -1
        dest_id = 0
        prev_router_id = 0
        prev_action = 0
        queueing_delay = 0

        while(True):
            #print("here")
            with open('input.txt','r') as f:
                lines = f.readlines()
                if(len(lines) == 7):
                    break

        my_id = int(lines[0])
        src_id = int(lines[1])
        dest_id = int(lines[2])
        prev_router_id = int(lines[3])
        prev_action = int(lines[4])
        queueing_delay = int(lines[5])
        cur_tick = int(lines[6])

#        print('Current Router ID:', my_id)
#        print('Destination Router ID:', dest_id)
        print('Epsilon:', EPSILON)
#        print(prev_router_id)
#        print(prev_action)
#        print('Queueing Delay:', queueing_delay)
        print('Current Tick:', cur_tick)
#        print('ITERATION:', ITERATIONS)
        os.remove("input.txt")
        
        if np.random.random() > EPSILON:
            m, action = get_qs(m, my_id, dest_id)
        else:
            action = np.random.randint(0, NACTIONS)

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

        if my_id != dest_id:
            outf = open("pythonaction.txt", "a")
            outf.write(str(action) + "\n")
            outf.close()
        

        done = 0
        if(my_id == dest_id):
            done = 1
            destcount += 1
        if my_id != src_id:
            rm = update_replay_memory(rm, my_id, dest_id, prev_router_id, prev_action, queueing_delay, done)

        if my_id == dest_id:
            #print('Inside python train function')
            m, tm, rm, tuc = train(m, tm, rm, tuc, my_id, dest_id)
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(MIN_EPSILON, EPSILON)

        #print('Action from Python:', action)

        if cur_tick == -100000:
            m.save('./saved_model')
            tm.save('./saved_target_model')
            with open('REPLAYMEM', 'wb') as f:
                pickle.dump(rm, f)
            with open('VARIABLES', 'wb') as f:
                pickle.dump(tuc, f)
            EPISODES += 1
            print('Extra lines:', destcount)            

#        if ITERATIONS == 30:
            #exit(0)
        print('----------------------')
