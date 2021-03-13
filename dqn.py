from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # neural network
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        # Returns Q-Values, indexed by actions
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            # self(input) calls the forward function
            # output = argmax(self(input))
            action = torch.argmax(self(state))
        else:
            # self.env.action_space.n represents possible actions
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    #state = Variable(torch.FloatTensor(np.float32(state)))
    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here
    loss = 0
    for x in range(batch_size):
        target_val = reward.data[x] + gamma * torch.max(target_model(next_state).data[x])
        model_val = reward.data[x] + gamma * torch.max(model(state).data[x])
        loss += (target_val - model_val)**2
        print(f'target model: {target_val}')
        print(f'model: {model_val}')
        print(f'loss: {loss}')
        
    return Variable(torch.FloatTensor([loss/batch_size]), requires_grad=True)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        # use random.sample
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        samples = random.sample(self.buffer, batch_size)
        for num in range(batch_size):
            state.append(samples[num][0])
            action.append(samples[num][1])
            reward.append(samples[num][2])
            next_state.append(samples[num][3])
            done.append(samples[num][4])
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
