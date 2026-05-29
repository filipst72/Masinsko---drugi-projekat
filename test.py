import torch as th
import numpy as np
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

device = 'cpu'

"""
if th.cuda.is_available():
    device = "cuda"
elif th.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
"""
LOCAL = True

#################### TO-DO ####################
class NN:
    def __init__(self, input_size=108, hidden_size=128, output_size=7, device='cpu'):
        self.device = device
        self.W1 = th.zeros(input_size, hidden_size, device=device)
        self.W2 = th.zeros(hidden_size, output_size, device=device)
        self.b1 = th.zeros(hidden_size, device=device)
        self.b2 = th.zeros(output_size, device=device)

    def relu(self, x):
        return th.maximum(th.tensor(0.0, device=self.device), x)

    def softmax(self, x):
        exp_x = th.exp(x - th.max(x, dim=1, keepdim=True)[0])
        return exp_x / exp_x.sum(dim=1, keepdim=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return a2

    def __call__(self, X):
        return self.forward(X)

model = NN(input_size=108, hidden_size=128, output_size=7, device=device)

saved_weights = th.load('bc_model.pth', map_location=device)
model.W1 = saved_weights['W1']
model.W2 = saved_weights['W2']
model.b1 = saved_weights['b1']
model.b2 = saved_weights['b2']

print("Successfully loaded model parameters from 'bc_model.pth'!")
###############################################


env = gym.make(
    "MiniGrid-Empty-Random-6x6-v0",
    render_mode="human" if LOCAL else "rgb_array",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

rewards = []

for episode in range(10):
    if model is None:
        break

    obs, _ = env.reset()
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step < 30:
        if LOCAL:
            env.render()
        with th.no_grad():
            obs = th.tensor(obs, dtype=th.float32, device=device).reshape(-1, 108)  #! promeniti po potrebi
            action = model(obs)
        obs, reward, terminated, truncated, _ = env.step(th.argmax(action).item())
        step += 1
    
    print(f"{episode=} {reward=}")
    rewards.append(reward)

env.close()
print(f"mean reward: {np.mean(rewards)}")
