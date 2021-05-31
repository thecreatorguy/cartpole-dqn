import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import gym

#! important questions:
# what is "kernel_initializer"? it appears to have the "he" name from before, is that how it initializes the weights?
# how does Sequential.compile work?
# tensor[None] means what? it doesn't appear to be the same as duplicating the full range with colon (:)
# what is the purpose of the "target network" here?
# 

class DqnAgent:

    @staticmethod
    def _build_dqn_model():
        q_net = tf.keras.Sequential()
        q_net.add(tf.keras.layers.Dense(64, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(tf.keras.layers.Dense(2, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')

    def __init__(self):
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    def policy(self, state):
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def train(self, batch):
        loss = 0
        return loss

class ReplayBuffer:

    def store_gameplay_experience(self, state, next_state, reward, action, done):
        return

    def sample_gameplay_batch(self):
        batch = []
        return batch


def collect_gameplay_experience(env, agent, buffer):
    state = env.reset()
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        buffer.store_gameplay_experience(state, next_state, reward, action, done)
        state = next_state

def train_model():
    env = gym.make('CartPole-v0')
    agent = DqnAgent()
    buffer = ReplayBuffer()

    for episode_cnt in range(6000):
        collect_gameplay_experience(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        if episode_cnt % 20 == 0:
            agent.update_target_network()


def main():
    train_model()


if __name__ == '__main__':
    main()