# PyTorch imports
import os

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx2pytorch import ConvertModel

# Environment import and set logger level to display error only
import gym
from gym import logger as gymlogger

from pyvirtualdisplay import Display

gymlogger.set_level(40)  # error only
pydisplay = Display(visible=0, size=(640, 480))
pydisplay.start()

# Seed random number generators
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None

action_mapping = [
    (0, 0, 0),  # no action
    (0, 0.5, 0),  # half throttle
    (0, 1, 0),  # full trottle
    (0, 0, 0.5),  # half break
    (0, 0, 1),  # full break
    # steering left with throttle/break control
    (-0.5, 0, 0),  # half left
    (-1, 0, 0),  # full left
    (-0.5, 0.5, 0),  # half left
    (-1, 0.5, 0),  # full left
    (-0.5, 1, 0),  # half left
    (-1, 1, 0),  # full left
    (-0.5, 0, 0.5),  # half left
    (-1, 0, 0.5),  # full left
    (-0.5, 0, 1),  # half left
    (-1, 0, 1),  # full left
    # steering right with throttle/break control
    (0.5, 0, 0),  # half right
    (1, 0, 0),  # full right
    (0.5, 0.5, 0),  # half right
    (1, 0.5, 0),  # full right
    (0.5, 1, 0),  # half right
    (1, 1, 0),  # full right
    (0.5, 0, 0.5),  # half right
    (1, 0, 0.5),  # full right
    (0.5, 0, 1),  # half right
    (1, 0, 1)  # full right
]


# Convert RBG image to grayscale and normalize by data statistics
def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, img_stack, seed=None):

        self.gym_env = gym.make('CarRacing-v0')
        self.env = self.gym_env
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        if seed is not None:
            self.env.seed(seed)


    def reset(self, raw_state=False):
        self.env = self.gym_env
        self.rewards = []
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        if raw_state:
            return np.array(self.stack), np.array(img_rgb)
        else:
            return np.array(self.stack)

    def step(self, action, raw_state=False):
        # for i in range(self.img_stack):
        img_rgb, reward, done, _ = self.env.step(action)
        # accumulate reward
        self.rewards.append(reward)
        # if no reward recently, end the episode
        die = True if np.mean(self.rewards[-np.minimum(100, len(self.rewards)):]) <= -1 else False
        if done or die:
            self.env.close()
        img_gray = rgb2gray(img_rgb)
        # add to frame stack
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        # --
        if raw_state:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die, img_rgb
        else:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die

    def close(self):
        self.env.close()



class Agent():
    """
    Agent for training
    """
    def __init__(self, net):
        self.net = net
        self.eval_eps = 0.001
        self.threshold = 0.3

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                net_response = self.net(state)
                q = net_response[0]
                logits = net_response[1]
                # Retrieve probabilities by exponentiating
                probs = F.softmax(logits, dim=1)
                # Create mask to exclude unlikely probabilities
                mask = (probs/probs.max(1, keepdim=True)[0] > self.threshold).float()
                # Use large negative number to mask actions away and select the next action greedily
                next_action_idx = int((mask * q + (1. - mask) * -1e8).argmax(1))
                return action_mapping[next_action_idx]
        else:
            # Randomly select an action uniformly
            next_action_idx = np.random.randint(len(action_mapping))
            return action_mapping[next_action_idx]


def run_episode(agent, img_stack, seed=None):
    env = Env(img_stack=img_stack, seed=seed)
    state = env.reset()
    score = 0
    done_or_die = False
    while not done_or_die:
        action = agent.select_action(state)
        state, reward, done, die = env.step(action)
        score += reward

        if done or die:
            done_or_die = True
    env.close()

    return score


if __name__ == "__main__":
    N_EPISODES = 50
    IMG_STACK = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Network
    net = ConvertModel(onnx.load(model_file))
    net = net.to(device)
    net.eval()
    agent = Agent(net)

    scores = []
    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, IMG_STACK, seed=seed))

    print(np.mean(scores))

