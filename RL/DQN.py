import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from collections import deque
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, state_space, h1, h2, action_space):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class DQN:
    def __init__(self, envir='CartPole-v0',
                 learning_rate=0.0001,
                 gamma=0.9,
                 initial_epsilon=0.5,
                 final_epsilon=0.01,
                 replay_size=10000,
                 batch_size=32):
        # environment
        self.env = gym.make(envir)
        self.relay_buffer = deque()
        self.time_step = 0
        # hyper parameters
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.relay_size = replay_size
        self.batch_size = batch_size
        self.gamma = gamma
        # create network
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.Q_value = Net(self.state_dim, 20, 20, self.action_dim)
        self.optimizer = torch.optim.Adam(self.Q_value.parameters(), lr=learning_rate)

    def saver(self, state, action, reward, next_state, done):
        """if the buffer is larger than the minibatch then start to train, and
        when the buffer size is larger than the relay size, remove the oldest examples"""
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.relay_buffer.append(
            (state, one_hot_action, reward, next_state, done))
        if len(self.relay_buffer) > self.relay_size:
            self.relay_buffer.popleft()
        if len(self.relay_buffer) > self.batch_size:
            self.train_Q_network()

    def greedy_selection(self, state):
        """
        greedy select if the randam generated value larger smaller than the greedy then random choose,
        else choose the max prob one
        """
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.Q_value(state)
        self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 10000
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            probs = probs.detach()
            probs = probs.data.numpy().astype(float)
            return np.argmax(probs)

    def train_Q_network(self):
        self.time_step += 1
        # random choose examples from the buffer
        minibatch = random.sample(self.relay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        Q_value_batch = [torch.reshape(self.Q_value(Variable(torch.from_numpy(x).float())),(1,2)) for x in next_state_batch]
        Q_value_batch = torch.cat(Q_value_batch)

        action_batch = [torch.reshape(torch.from_numpy(x).float(),(1,2)) for x in action_batch]
        action_batch = torch.cat(action_batch,0)

        Q_exp = Q_value_batch * action_batch
        Q_exp = torch.sum(Q_exp,1)

        y_batch = []
        for i in range(0, self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i].data.numpy()))

        self.optimizer.zero_grad()
        y_batch = torch.from_numpy(np.array(y_batch))
        y_batch = y_batch.double()
        Q_exp = Q_exp.double()
        loss = torch.mean((y_batch - Q_exp)**2)
        loss.backward()
        self.optimizer.step()

    def run(self, episodes):
        cycle = []
        for e in range(episodes):
            state = self.env.reset()
            for step in range(300):
                action = self.greedy_selection(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = -1 if done else 0.1
                self.saver(state, action, reward, next_state, done)
                state = next_state
                if done:
                    cycle.append(step)
                    self.plot_durations(cycle)
                    break

    def plot_durations(self, episode_durations):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)


if __name__ == "__main__":
    dqn = DQN()
    dqn.run(1000)
