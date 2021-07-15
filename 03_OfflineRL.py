import copy
import random
from time import sleep

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from gym import logger as gymlogger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import config
from auxiliary_methods import Env
from auxiliary_methods import Logger
from auxiliary_methods import action_mapping
from auxiliary_methods import plot_metrics
from auxiliary_methods import save_as_onnx
from bcq import QNet
from dataset import DatasetDemonstration
from dataset import PartialDataset

sns.set()
gymlogger.set_level(40)  # error only

seed = config.seed
if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


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

        self.Q = QNet(img_stack, self.num_actions, use_weights_init=True).to(device)
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
        q_vals, logits = self.Q(state)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        # probs = (probs / probs.max(1, keepdim=True)[0] > self.threshold).float()
        # action_idx = int((probs * q_vals + (1. - probs) * -1e8).argmax(1))
        probs = (probs / probs.max(1, keepdim=True)[0] > self.threshold).float()
        action_idx = (probs * q_vals + (1 - probs) * -1e8).argmax(1, keepdim=True)

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
            # next_action = [action_mapping[idx] for idx in action_idx.squeeze().tolist()]
            # Get target q-function
            q, _ = self.Q_target(next_state)
            target_Q = reward + done * self.discount * q.gather(1, action_idx).reshape(-1, 1)

        # Get current Q estimate
        current_Q, logits = self.Q(state)
        log_probs = F.log_softmax(logits, dim=1)
        # Gather actions along dimension
        current_Q = current_Q.gather(1, action)
        # Get log probabilities from logits

        # computes losses
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        policy_loss = F.nll_loss(log_probs, action.reshape(-1))
        loss = q_loss + policy_loss + 1e-3 * logits.pow(2).mean()

        self.take_gradient_step(loss)

        self.update_target_by_polyak()

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
        # if use_colab_autodownload: download_colab_model(param_file)

    def load_param(self, param_file):
        self.Q.load_state_dict(torch.load(param_file, map_location="cpu"))

    def take_gradient_step(self, loss):
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

    def update_target_by_polyak(self):
        for Q_param, Q_target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            Q_target_param.data.copy_(self.tau * Q_param.data + (1 - self.tau) * Q_target_param)


def train_epoch(agent, train_set, logger, epoch, pbar, epochs, batchsize):
    # Switch to train mode
    agent.mode('train')
    # Initialize helpers variables
    ts_len = len(train_set)
    running_loss = None
    alpha = 0.3
    for i, idx in enumerate(BatchSampler(SubsetRandomSampler(range(ts_len)), 1, False)):
        # data = train_set.load_multiple_files(idx)
        data = train_set.load(idx[0])
        partial = PartialDataset(data)
        loader = DataLoader(partial, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=False,
                            pin_memory=True)
        l_len = len(loader)
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


@torch.no_grad()
def run_episode(agent, n_runs=1, record_video=False, logger=None, pbar=None, img_stack=config.img_stack):
    agent.mode('eval')
    score_avg = None
    alpha = 0.3
    env_seeds = [np.random.randint(1e7) for _ in range(n_runs)]
    for i in range(n_runs):
        # Create new environment object
        env = Env(img_stack=img_stack, record_video=record_video, seed=env_seeds[i])
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


# Specify the google drive mount here if you want to store logs and weights there (and set it up earlier)
logger = Logger("logdir")
print("Saving state to {}".format(logger.basepath))


def training(pretrained=False, filepath_pretrained="model.pkl", epochs=2, img_stack=4):
    writer = config.tensorboard
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

            train_loss, sample = train_epoch(agent, train_set, logger, i_ep, pbar, epochs, config.batchsize)
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
            writer.add_scalar('Loss_train', train_loss, i_ep)
            writer.add_scalar('Reward', score, i_ep)
            print("Saving state to {}".format(logger.basepath))
            agent.save("model.pkl", sample)

    print("Saved state to {}".format(logger.basepath))
    print("[%03d] Training Loss: %.4f" % (i_ep + 1, train_loss))
    plot_metrics(logger)


training(img_stack=config.img_stack, epochs=config.epochs, pretrained=True)

# # Visualize Agent Interactions
# ### Put the agent into a real environment
# Let's see how the agent is doing in the real environment


# select agent you want to evaluate
agent = ORLAgent(logger, img_stack=config.img_stack)

# load parameter
# param_file = 'logdir/2021-03-29T09-24-10/params.pkl_009'
# agent.load_param(param_file)

# run episode with recording and show video
run_episode(agent, n_runs=1, record_video=True)
# show_video(env.video_dir)
