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
import sys
import pickle
from os import path

REPLAYMEM = "replay.dat"
VARIABLES = "variable.dat"
NACTIONS = 4
GRIDSIZE = 8
NROUTERS = GRIDSIZE * GRIDSIZE
NSTATES = 2 * NROUTERS # Number of states in case of one-hot encoding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

LEARNINGRATE = 0.01
DISCOUNT = 0.9
EPSILON = 0.99
EPSILON_DECAY = 0.9992
MIN_EPSILON = 0.01
REWARD_FACTOR = 100000

REPLAY_MEMORY_SIZE = 2000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 64  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 64  # Terminal states (end of episodes)

# Try using ordinal encoding as there is relationship between inputs
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

    # An array with last n steps for training
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    # Used to count when to update target network with main network's weights
    target_update_counter = 0

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
def update_replay_memory(replay_memory, my_id, dest_id, prev_router_id, prev_action, queueing_delay, done, is_dead):

    transition = [my_id, dest_id, prev_router_id, prev_action, queueing_delay, done, is_dead]
    replay_memory.append(transition)

    return replay_memory


# Trains main network every step during episode
def train(model, target_model, replay_memory, target_update_counter):

    # Start training only if certain number of samples is already saved
    if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return model, target_model, replay_memory, target_update_counter

    # Get a minibatch of random samples from memory replay table
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

    # Get current states from minibatch, then query NN model for Q values
    current_states = np.array([oneHotEncode(transition[2], transition[1]) for transition in minibatch])
    current_qs_list = model.predict(current_states)

    current_dead_states = np.array([oneHotEncode(transition[0], transition[1]) for transition in minibatch])
    current_qs_dead_list = model.predict(current_dead_states)    

    # Get future states from minibatch, then query NN model for Q values
    # When using target network, query it, otherwise main network should be queried
    new_current_states = np.array([oneHotEncode(transition[0], transition[1]) for transition in minibatch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    y = []

    # Now we need to enumerate our batches
    # reward here is queuing delay
    for index, (my_id, dest_id, prev_router_id, prev_action, reward, done, is_dead) in enumerate(minibatch):

        if not is_dead:
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = REWARD_FACTOR / reward + DISCOUNT * max_future_q
            else:
                new_q = REWARD_FACTOR / reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[prev_action] = new_q

            # And append to our training data
            X.append(oneHotEncode(prev_router_id, dest_id))
            y.append(current_qs)
        else:
            new_q = -10

            current_qs = current_qs_dead_list[index]
            current_qs[prev_action] = new_q

            # And append to our training data
            X.append(oneHotEncode(my_id, dest_id))
            y.append(current_qs)

    # Fit on all samples as one batch, log only on terminal state
    model.fit(np.array(X), np.array(y), batch_size = MINIBATCH_SIZE, verbose=0)

    if done:
        target_update_counter += 1

    # If counter reaches set value, update target network with weights of main network
    if target_update_counter > UPDATE_TARGET_EVERY:
        target_model.set_weights(model.get_weights())
        target_update_counter = 0
    
    return model, target_model, replay_memory, target_update_counter



# Queries main network for Q values given current observation space (environment state)
def get_qs(model, my_id, dest_id):

    state = oneHotEncode(my_id, dest_id)
    actions = model.predict(np.array(state).reshape(-1, NSTATES))
    optimal_action = np.argmax(actions)
    
    return model, optimal_action


print("\nSTARTING DQN PYTHON EXECUTION!")


if __name__ == "__main__":

    reset_variables = sys.argv[1] if len(sys.argv) > 1 else 0
    iterations = sys.argv[2] if len(sys.argv) > 2 else 0

    m = None
    tm = None
    rm = None
    tuc = None

    if reset_variables:
        m, tm, rm, tuc = initialize()
    else:
        m = tf.keras.models.load_model('./saved_model')
        tm = tf.keras.models.load_model('./saved_target_model')
        with open('REPLAYMEM', 'rb') as f:
            rm = pickle.load(f)
        with open('VARIABLES', 'rb') as f:
            tuc = pickle.load(f)

    while(1):
        while(1):
            if(path.exists("input.txt")):
                break

        my_id = -1
        dest_id = 0
        prev_router_id = 0
        prev_action = 0
        queueing_delay = 0

        while(True):
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

        os.remove("input.txt")
        
        iterations += 1
        print('Iteration: ', iterations, '   Epsilon:', EPSILON)

        done = 0
        if(my_id == dest_id):
            done = 1

        my_x = my_id % GRIDSIZE
        my_y = my_id // GRIDSIZE

        is_dead = False
        generate_model_action = np.random.random() > EPSILON
        if generate_model_action:
            m, action = get_qs(m, my_id, dest_id)

            if(action == 0 and my_y == GRIDSIZE-1):
                is_dead = True
            elif(action == 1 and my_x == GRIDSIZE-1):
                is_dead = True
            elif(action == 2 and my_y == 0):
                is_dead = True
            elif(action == 3 and my_x == 0):
                is_dead = True
            
            rm = update_replay_memory(rm, my_id, dest_id, prev_router_id, action, queueing_delay, done, is_dead)

            
        if is_dead or not generate_model_action:
            is_dead = False
            action = np.random.randint(0, NACTIONS)
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
                    action = np.random.randint(0, NACTIONS)

        f = open("action.txt","w")
        f.write(str(action))
        f.close()


        if my_id != src_id:
            rm = update_replay_memory(rm, my_id, dest_id, prev_router_id, prev_action, queueing_delay, done, is_dead)

        if my_id == dest_id:
            m, tm, rm, tuc = train(m, tm, rm, tuc)

        if iterations % 50 == 0:
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(MIN_EPSILON, EPSILON)
            if iterations == 200000:
                EPSILON = MIN_EPSILON


        if cur_tick == 100000:
            m.save('./saved_model')
            tm.save('./saved_target_model')
            with open('REPLAYMEM', 'wb') as f:
                pickle.dump(rm, f)
            with open('VARIABLES', 'wb') as f:
                pickle.dump(tuc, f)


