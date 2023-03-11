import gymnasium as gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from FrameStack import FrameStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
learning_rate = 0.000025
gamma         = 0.99
buffer_limit  = 80000
batch_size    = 32


transform = transforms.Compose([
    transforms.ToPILImage(),        # PyTorch 텐서를 PIL 이미지로 변환
    transforms.Resize((84, 84)),   # 이미지 크기를 84x84로 조정
    transforms.Grayscale(),        # 이미지를 흑백으로 변환
    transforms.ToTensor(),         # 이미지를 텐서로 변환
])

# observation 전처리 함수
def preprocess(observation):
    # observation을 PyTorch 텐서로 변환
    observation = np.array(observation)
    # transform을 이용하여 observation을 전처리
    observation = transform(observation)
    # 전처리된 observation을 반환
    return observation

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
            
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.stack(s_lst).to(device), torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device), torch.stack(s_prime_lst).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU()
		)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU()
		)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
		)
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
		)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 3136)
        x = self.fc(x)
        return x
      
    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else : 
            out = self.forward(obs.to(device))
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    q_out = q(s)
    q_out = q_out.to(device)
    q_a = q_out.gather(1,a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    
    # q = Qnet().to(device)
    q = torch.load('q.pth')
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    s_stack = FrameStack(4)
    s_prime_stack = FrameStack(4)

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(100000):
        # epsilon = max(0.1,  0.15 - 0.01*(n_epi/200)) #Linear annealing from 8% to 5%
        epsilon = 0
        s = env.reset()
        s = Image.fromarray(s[0])
        s = preprocess(s)
        tmp_stack = s_stack.Get()
        done = False
        while not done:
            for i in range(20):
                a = q.sample_action(tmp_stack, epsilon)
                s_prime, r, terminated, truncated, info = env.step(a)
                s_prime = Image.fromarray(s_prime)
                s_prime = preprocess(s_prime)
                s_stack.put(s)
                s_prime_stack.put(s_prime)
                done_mask = 0.0 if terminated  else 1.0
                memory.put((s_stack.Get(), a, r, s_prime_stack.Get(), done_mask))
                s = s_prime
                tmp_stack = s_prime_stack.Get()
                score += r
                
            if memory.size() > 50000:
                train(q, q_target, memory, optimizer)
                
            if terminated:
                break
             
        if (n_epi % 2 == 0 and n_epi != 0):
            q_target.load_state_dict(q.state_dict())
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()