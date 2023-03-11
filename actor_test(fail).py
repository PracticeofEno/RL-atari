import math
import random
import gymnasium as gym
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from actor import ActorCritic

env = gym.make("ALE/Breakout-v5", render_mode="human")

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

# observation 전처리
# obs = Image.fromarray(observation)
# if obs.mode != 'RGB':
#     obs = obs.convert('RGB')
# observation = preprocess(observation[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(device).to(device)

epsilon = 0.9  # 초기 엡실론 값
epsilon_min = 0.01  # 최소 엡실론 값
epsilon_decay = 0.999  # 엡실론 감소 비율

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 50

score = 0
steps_done = 0
for i in range(num_episodes):
    # Initialize the environment and get it's state
    episode_end = True
    s = env.reset()
    s = Image.fromarray(s[0])
    s = preprocess(s).to(device)
    score = 0
    while episode_end:
        for t in range(20):
            sample = random.random()
            # 에피소드를 진행할수록 랜덤하게 뽑지않고 최적값을 기준으로 액션을 하게 만들기 위한 변수
            eps_threshold = max(epsilon * epsilon_decay, epsilon_min)
            
            if sample > eps_threshold:
                prob = model.pi(s, 1)
                a = prob.max(1)[1]
            else:
                a = random.randint(0, 3)
            s_prime, r, terminated, truncated, info = env.step(a)
            episode_end = not terminated
            s_prime = Image.fromarray(s_prime)
            s_prime = preprocess(s_prime).to(device)
            model.put_data((s,a,r,s_prime,terminated))
            
            s = s_prime
            score += r
            if terminated:
                break
        model.train_net()
        steps_done += 1
    epsilon = epsilon * epsilon_decay
    if i%20==0 and i!=0:
        print("# of episode :{}, avg reward : {:.4f}".format(i, score/20))
        score = 0

env.close()