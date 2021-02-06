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


REPLAYMEM = "replay.dat"
VARIABLES = "variable.dat"
NACTIONS = 4
NROUTERS = 16
NSTATES = 2 * NROUTERS # Number of states - 32
GRIDSIZE = 4

LEARNINGRATE = 0.001
DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


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

    model.save('./saved_model')
    target_model.save('./saved_target_model')
    # An array with last n steps for training
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    with open(REPLAYMEM, "wb") as f:
        pickle.dump(replay_memory, f)

    # Used to count when to update target network with main network's weights
    target_update_counter = 0

    with open(VARIABLES, "wb") as f:
        pickle.dump(target_update_counter, f)


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
def update_replay_memory(my_id, dest_id, prev_router_id, prev_action, queueing_delay, done):
    replay_memory = []
    with open(REPLAYMEM, "rb") as f:
        replay_memory = pickle.load(f)

    transition = [my_id, dest_id, prev_router_id, prev_action, queueing_delay, done]
    replay_memory.append(transition)

    with open(REPLAYMEM, "wb") as f:
        pickle.dump(replay_memory, f)


# Trains main network every step during episode
def train(my_id, dest_id):

    replay_memory = []
    with open(REPLAYMEM, "rb") as f:
        replay_memory = pickle.load(f)

    # Start training only if certain number of samples is already saved
    if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return

    model = tf.keras.models.load_model('./saved_model')
    target_model = tf.keras.models.load_model('/saved_target_model')
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

    # save and load target_update_counter
    # Update target network counter every episode
    target_update_counter = 0
    with open(VARIABLES, "rb") as f:
        target_update_counter = pickle.load(f)

    if done:
        target_update_counter += 1
        with open(VARIABLES, "wb") as f:
            pickle.dump(target_update_counter, f)

    # If counter reaches set value, update target network with weights of main network
    if target_update_counter > UPDATE_TARGET_EVERY:
        target_model.set_weights(model.get_weights())
        target_update_counter = 0
        target_model.save('./saved_target_model')
    
    model.save('./saved_model')
    

# Queries main network for Q values given current observation space (environment state)
def get_qs(my_id, dest_id):
    model = tf.keras.models.load_model('./saved_model')
    state = oneHotEncode(my_id, dest_id)
    actions = model.predict(np.array(state).reshape(-1, NROUTERS*2))
    optimal_action = np.argmax(actions)
    return optimal_action

