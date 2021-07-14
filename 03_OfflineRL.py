import copy
import math
import os
import random
import shutil
from collections import namedtuple
from time import sleep

import gym
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from gym import logger as gymlogger
from gym.wrappers import Monitor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from torch.nn.functional import log_softmax

from auxiliary_methods import Logger
from auxiliary_methods import plot_metrics
from auxiliary_methods import rgb2gray
from auxiliary_methods import hide_hud
from auxiliary_methods import show_video
from auxiliary_methods import action_mapping

sns.set()
gymlogger.set_level(40)  # error only

seed = 777
if seed:
    random.seed(seed)

if seed:
    np.random.seed(seed)

if seed:
    torch.manual_seed(seed)

# # Dataloader


# create transition object for partial demonstrations
Transition = namedtuple('Transition', ['states', 'actions', 'next_states', 'rewards', 'dones'])


# Since the demonstrations are partial files assuming that the collected data is too
# large to fit into memory at once the Demonstration class utilizes an object 
# from the ParialDataset class to load and unload files from the file system.
# This is a typical use case for very large datasets and should give you an idea 
# how to handle such issues.  
class DatasetDemonstration(object):
    def __init__(self, root_path):
        assert (os.path.exists(root_path))
        self.root_path = root_path
        # assign list of data files found in the data root directory
        self.data_files = sorted(os.listdir(root_path))

    def __len__(self):
        # this count returns the number of files in the data root folder
        return len(self.data_files)

    def load(self, idx):
        # select an index at random from all files
        file_name = self.data_files[idx]
        file_path = os.path.join(self.root_path, file_name)
        # load the selected file
        data = np.load(file_path)
        # get the respective properties from the files
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        rewards = data["rewards"]
        dones = data["dones"]
        # clean the memory from the data file
        del data
        # return the transitions
        return Transition(states=states, actions=actions, next_states=next_states, rewards=rewards, dones=dones)


# Itereates over the dataset subset in the known manner.
class PartialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.states)

    def __getitem__(self, idx):
        # select stack of states
        states = self.data.states[idx]
        # select followup action, next_state, reward and done flag
        action = self.data.actions[idx]
        next_state = self.data.next_states[idx]
        reward = self.data.rewards[idx]
        done = self.data.dones[idx]

        return states, action, next_state, reward, done


# # Inspect data
img_stack = 4
show_hud = True
batchsize = 128
use_colab_autodownload = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

activation_mapping = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "identity": nn.Identity()}


# # BCQ Implementation

# Defining a Q-Network for predicting and evaluating the next action given a state
class QNet(nn.Module):
    def __init__(self, img_stack, n_units_out, use_bias=True, use_weights_init=True, activation="relu"):
        super(QNet, self).__init__()
        self.n_units_in = img_stack
        self.n_units_out = n_units_out
        self.bias = use_bias
        self.activation = activation_mapping[activation]

        self.cnn_1 = nn.Conv2d(self.n_units_in, 32, kernel_size=8, stride=4)
        self.cnn_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.q_vals_l1 = nn.Linear(4096, 512)
        self.q_vals_l2 = nn.Linear(512, self.n_units_out)

        self.policy_output_l1 = nn.Linear(4096, 512)
        self.policy_output_l2 = nn.Linear(512, self.n_units_out)

        if use_weights_init:
            pass

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.activation(self.cnn_1(x))
        x = self.activation(self.cnn_2(x))
        x = self.activation(self.cnn_3(x))

        q_vals_l1 = self.activation(self.q_vals_l1(x.reshape(-1, 4096)))
        logits = self.policy_output_l2(self.activation(self.policy_output_l1(x.reshape(-1, 4096))))

        # returns q-function, log action probability and action logits
        return self.q_vals_l2(q_vals_l1), log_softmax(logits, dim=1), logits


# Agent for the Discrete Batch-Constrained deep Q-Learning (BCQ) https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
class ORLAgent(object):
    def __init__(
            self,
            logger,
            img_stack,  # image stack
            threshold=0.3,  # threshold to bias away actions
            eval_eps=0.001,  # action sampling epsilon
            discount=0.99,  # discount factor for Q-value computation
            lambda_=0.01,  # regularization parameter
            tau=0.01,  # parameter for exponential moving average of Q parameter updates
            optimizer="Adam",
            optimizer_parameters={"lr": 0.000625, "eps": 0.00015}):
        self.logger = logger
        self.num_actions = len(action_mapping)

        self.Q = QNet(img_stack, self.num_actions).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        self.tau = tau
        # Evaluation hyperparameter for action selection
        self.eval_eps = eval_eps
        # Threshold for biasing unlikely actions away
        self.threshold = threshold
        # Regularization hyperparameter
        self.lambda_ = lambda_

        num_trainable_params = sum(p.numel() for p in self.Q.parameters() if p.requires_grad)
        print("Trainable Parameters: {}".format(num_trainable_params))

    def _get_action_idx(self, state) -> int:
        # =========== YOUR CHANGES =============
        # ######################################
        # Get the action index based on the state provided
        # using the Q-Network
        # ######################################
        # 1) get current state q values and policy logits
        # 2) retriev probabilities through softmax
        # 3) create action mask using the threshold
        # 4) compute the next action index with the weighted q values and mask
        q_vals, probs, logits = self.Q(state)
        probs = probs.exp()
        probs = (probs / probs.max(1, keepdim=True)[0] > self.threshold).float()
        action_idx = int((probs * q_vals + (1. - probs) * -1e8).argmax(1))
        return action_idx

    def select_action(self, state):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > self.eval_eps:
            with torch.no_grad():
                next_action_idx = self._get_action_idx(state)
                return action_mapping[next_action_idx]
        else:
            # Randomly select an action uniformly
            next_action_idx = np.random.randint(self.num_actions)
            return action_mapping[next_action_idx]

    @torch.enable_grad()
    def train(self, state, action, next_state, reward, done):
        # Compute the target Q value
        with torch.no_grad():
            action_idx = self._get_action_idx(next_state)
            # Get target q-function
            q, _ = self.Q_target(next_state)
            # =========== YOUR CHANGES =============
            # ######################################
            # Calculate the target q values to 
            # update the Q-Network
            # ######################################
            # 1) compute the target Q-value
            target_Q = ...

        # Get current Q estimate
        current_Q, logits = self.Q(state)
        # Gather actions along dimension
        current_Q = current_Q.gather(1, action)
        # Get log probabilities from logits
        log_probs = self.log_softmax(logits)

        # =========== YOUR CHANGES =============
        # ######################################
        # Compute the loss based on the q values,
        # the policy constrain and an optional 
        # regularization.
        # ######################################
        # 1) compute Q loss using the smoothed L1 loss
        # 2) compute policy loss via the negative log-likelihood between log probabilites and demonstration actions
        # 3) regularize based on logits 
        # 4) compute total loss
        # 5) take a backward step on the Q function
        # 6) update target network by polyak by iterating over the Q-Network and target Q-Network parameters
        #    use tau parameter to compute the exponential moving average Q-Network parameters and update the target network

        # Return loss
        return loss.cpu().item()

    def mode(self, mode):
        # switch networks between evaluation and train mode
        if mode == 'train':
            self.Q.train()
        else:
            self.Q.eval()

    def save(self, param_file, sample):
        torch.save(self.Q.state_dict(), param_file)
        save_as_onnx(self.Q, sample, f'{param_file}_onnx')
        # download param file
        if use_colab_autodownload: download_colab_model(param_file)

    def load_param(self, param_file):
        self.Q.load_state_dict(torch.load(param_file, map_location="cpu"))

    # # Define Training and Validation Routines


def train_epoch(agent, train_set, logger, epoch, pbar, epochs):
    # Switch to train mode
    agent.mode('train')
    # Initialize helpers variables
    ts_len = 2 # len(train_set)
    running_loss = None
    alpha = 0.3
    # =========== [OPTIONAL] CHANGES =============
    # ############################################
    # [Hint]: Accessing the file system is slow and you can
    # reshape your data / load multiple files in to speed up 
    # training.
    # ############################################
    # Iterate over the list of demonstration files
    for i, idx in enumerate(BatchSampler(SubsetRandomSampler(range(ts_len)), 1, False)):
        # Load the selected index from the filesystem
        data = train_set.load(idx[0])
        # Create dataset from loaded data sub-set
        partial = PartialDataset(data)
        # Create dataloader
        loader = DataLoader(partial, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=False, pin_memory=True)
        l_len = len(loader)
        # Iterate over parial dataset
        for j, (s, a, s_, r, d) in enumerate(loader):
            # Adjust types, shape and push to device
            s = s.float().to(device)
            a = a.long().unsqueeze(1).to(device)
            s_ = s_.float().to(device)
            r = r.float().unsqueeze(1).to(device)
            d = d.float().unsqueeze(1).to(device)
            # Train the respective agent
            loss = agent.train(s, a, s_, r, d)
            # Update running average loss
            running_loss = loss if running_loss is None else loss * alpha + (1 - alpha) * running_loss
            # Update info in the progress bar
            pbar.set_postfix_str("Epoch: %03d/%03d Partial: %03d/%03d Idx: %03d/%03d Loss: %.4f" % (
                epoch + 1, epochs, i + 1, ts_len, j + 1, l_len, running_loss))
    return running_loss, s  # s serves as sample input for saving the model in ONNX format


# # Evaluate the agent in the real environment

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, img_stack, show_hud=True, record_video=True, seed=None):
        self.record_video = record_video
        # Create gym environment
        self.gym_env = gym.make('CarRacing-v0')
        if seed:
            print(f"Environment seed: {seed}")
            self.gym_env.seed(seed)
            self.gym_env.action_space.seed(seed)
        self.env, self.video_dir = self.wrap_env(self.gym_env)
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        self.show_hud = show_hud

    def reset(self, raw_state=False):
        self.env, self.video_dir = self.wrap_env(self.gym_env)
        self.rewards = []
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        if not self.show_hud:
            img_gray = hide_hud(img_gray)
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
        if not self.show_hud:
            img_gray = hide_hud(img_gray)
        # add to frame stack  
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        # --
        if raw_state:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die, img_rgb
        else:
            return np.array(self.stack), np.sum(self.rewards[-1]), done, die

    def render(self, *arg):
        return self.env.render(*arg)

    def close(self):
        self.env.close()

    def wrap_env(self, env):
        """
        Wrapper for recording video of the environment.
        """
        outdir = f"./videos/"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)
        if self.record_video:
            env = Monitor(env, outdir, force=True)
        return env, outdir


@torch.no_grad()
def run_episode(agent, n_runs=1, record_video=False, logger=None, pbar=None):
    agent.mode('eval')
    score_avg = None
    alpha = 0.3
    env_seeds = [np.random.randint(1e7) for _ in range(n_runs)]
    for i in range(n_runs):
        # Create new environment object
        env = Env(img_stack=img_stack, record_video=record_video, seed=env_seeds[i], show_hud=False)
        state = env.reset()
        done_or_die = False
        score = 0
        while not done_or_die:
            t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = agent.select_action(t_state)
            state, reward, done, die = env.step(action)
            score += reward
            if pbar:
                pbar.set_postfix_str("Env Evaluation - Run {:03d} Score: {:.2f}".format(i + 1, score))
            if done or die:
                done_or_die = True
            sleep(0.001)
        env.close()
        score_avg = score if score_avg is None else score * alpha + (1 - alpha) * score_avg
        print(f"Evaluation run {i} completed!")
    return score_avg


# # Train your agent


# Specify the google drive mount here if you want to store logs and weights there (and set it up earlier)
logger = Logger("logdir")
print("Saving state to {}".format(logger.basepath))


def training(pretrained=False, filepath_pretrained="pretrained.pkl", epochs = 2):
    train_set = DatasetDemonstration('data-mixed')
    agent = ORLAgent(logger, img_stack=img_stack)
    if pretrained:
        agent.load_param(filepath_pretrained)

    epoch_iter = range(epochs)
    with tqdm(epoch_iter) as pbar:
        for i_ep in pbar:
            print(f"Starting training epoch {i_ep + 1}/{epochs}")
            # plot current training state
            if i_ep > 0:
                plot_metrics(logger)

            train_loss, sample = train_epoch(agent, train_set, logger, i_ep, pbar, epochs)
            logger.log("training_loss", train_loss)
            # =========== [OPTIONAL] CHANGES =============
            # ############################################
            # Go full Offline RL if you feel up to it. :)
            # [Hint]: Evaluate in the environment - strictly speaking this is not allowed in pure Offline RL!!!
            # But we ease the task a bit and avoid that you are flying blind all the time.
            # Otherwise you would be only allowed to test once you submit to the challenge server.
            # If you are really looking for a challenge feel free to remove this line and make a train/eval data split from the demonstrations.
            # ############################################
            score = run_episode(agent, logger=logger, pbar=pbar)
            logger.log("env_score", score)
            # store logs
            logger.dump()
            # store weights
            print("Saving state to {}".format(logger.basepath))
            save_file_path = f'{logger.param_file}_%03d' % i_ep
            agent.save(save_file_path, sample)

    print("Saved state to {}".format(logger.basepath))
    print("[%03d] Training Loss: %.4f" % (i_ep + 1, train_loss))
    plot_metrics(logger)


training()

# # Visualize Agent Interactions
# ### Put the agent into a real environment
# Let's see how the agent is doing in the real environment


# select agent you want to evaluate
agent = ORLAgent(logger, img_stack=img_stack)

# load parameter
# param_file = 'logdir/2021-03-29T09-24-10/params.pkl_009'
# agent.load_param(param_file)

# run episode with recording and show video
run_episode(agent, n_runs=1, record_video=True)
# show_video(env.video_dir)
