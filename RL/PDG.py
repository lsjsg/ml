import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym


class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class PolicyGD:
    def __init__(self, envir='CartPole-v0', learning_rate=0.01, gamma=0.99, batch_size=5, state_space=4, action_space=1):
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = gym.make(envir)
        self.net = Net(state_space, action_space)
        self.optimizer = torch.optim.RMSprop(
            self.net.parameters(), lr=learning_rate)

    def main(self, num_episode=5000):
        episode_durations = []
        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
        for e in range(num_episode):
            state = self.env.reset()
            state = torch.from_numpy(state).float()
            state = Variable(state)
            for t in count():
                probs = self.net(state)
                m = Bernoulli(probs)
                action = m.sample()
                action = action.data.numpy().astype(int)[0]
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = 0

                state_pool.append(state)
                action_pool.append(float(action))
                reward_pool.append(reward)

                state = next_state
                state = torch.from_numpy(state).float()
                state = Variable(state)
                steps += 1

                if done:
                    episode_durations.append(t + 1)
                    self.plot_durations(episode_durations)
                    break

            if e > 0 and e % self.batch_size == 0:
                self.update(state_pool, action_pool, reward_pool, steps)
                state_pool = []
                action_pool = []
                reward_pool = []
                steps = 0

    def update(self, state_pool, action_pool, reward_pool, steps):
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(steps):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        self.optimizer.zero_grad()

        for i in range(steps):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            probs = self.net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()
        self.optimizer.step()

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


if __name__ == '__main__':
    rl = PolicyGD(state_space=4, action_space=1)
    rl.main()
