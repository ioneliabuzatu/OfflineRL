import base64
import glob
import io
import os
import random
import shutil
import time
from time import time, strftime

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
import torch
from gym import logger as gymlogger
from gym.wrappers import Monitor
import config
sns.set()
gymlogger.set_level(40)  # error only
seed=config.seed
if seed:
    if seed: random.seed(seed)
    if seed: np.random.seed(seed)
    if seed: torch.manual_seed(seed)

# Action space (map from continuous actions for steering, throttle and break to 25 action combinations)
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


# # Auxiliary Methods
class Logger():
    def __init__(self, logdir, params=None):
        self.basepath = os.path.join(logdir, strftime("%Y-%m-%dT%H-%M-%S"))
        os.makedirs(self.basepath, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if params is not None and os.path.exists(params):
            shutil.copyfile(params, os.path.join(self.basepath, "params.pkl"))
        self.log_dict = {}
        self.dump_idx = {}

    @property
    def param_file(self):
        return os.path.join(self.basepath, "params.pkl")

    @property
    def onnx_file(self):
        return os.path.join(self.basepath, "model.onnx")

    @property
    def log_dir(self):
        return os.path.join(self.basepath, "logs")

    def log(self, name, value):
        if name not in self.log_dict:
            self.log_dict[name] = []
            self.dump_idx[name] = -1
        self.log_dict[name].append((len(self.log_dict[name]), time(), value))

    def get_values(self, name):
        if name in self.log_dict:
            return [x[2] for x in self.log_dict[name]]
        return None

    def dump(self):
        for name, rows in self.log_dict.items():
            with open(os.path.join(self.log_dir, name + ".log"), "a") as f:
                for i, row in enumerate(rows):
                    if i > self.dump_idx[name]:
                        f.write(",".join([str(x) for x in row]) + "\n")
                        self.dump_idx[name] = i


def plot_metrics(logger):
    train_loss = logger.get_values("training_loss")
    env_score = logger.get_values("env_score")

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, label="train")
    ax2 = fig.add_subplot(132, label="score")

    ax1.plot(train_loss, color="C0")
    ax1.set_ylabel("Loss", color="black")
    ax1.set_xlabel("Epoch", color="black")
    ax1.tick_params(axis='x', colors="black")
    ax1.tick_params(axis='y', colors="black")
    ax1.set_ylim((0, 10))

    ax2.plot(env_score, color="C1")
    ax2.set_ylabel("Score", color="black")
    ax2.set_xlabel("Epoch", color="black")
    ax2.tick_params(axis='x', colors="black")
    ax2.tick_params(axis='y', colors="black")
    ax2.set_ylim((-100, 1000))

    fig.tight_layout(pad=2.0)
    plt.show()


def print_action(action):
    print("Left %.1f" % action[0] if action[0] < 0 else "Right %.1f" % action[0] if action[0] > 0 else "Straight")
    print("Throttle %.1f" % action[1])
    print("Break %.1f" % action[2])


"""
Utility functions to enable video recording of gym environment and displaying it
"""


def concatenate_videos(video_dir):
    """
    Merge all mp4 videos in video_dir.
    """
    outfile = os.path.join(video_dir, 'merged_video.mp4')
    cmd = "ffmpeg -i \"concat:"
    mp4list = glob.glob(os.path.join(video_dir, '*.mp4'))
    tmpfiles = []
    # build ffmpeg command and create temp files
    for f in mp4list:
        file = os.path.join(video_dir, "temp" + str(mp4list.index(f) + 1) + ".ts")
        os.system("ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
        tmpfiles.append(file)
    for f in tmpfiles:
        cmd += f
        if tmpfiles.index(f) != len(tmpfiles) - 1:
            cmd += "|"
        else:
            cmd += f"\" -c copy  -bsf:a aac_adtstoasc {outfile}"
    # execute ffmpeg command to combine videos
    os.system(cmd)
    # cleanup
    for f in tmpfiles + mp4list:
        if f != outfile:
            os.remove(f)
    # --
    return outfile


def show_video(video_dir):
    """
    Show video in the output of a code cell.
    """
    # merge all videos
    mp4 = concatenate_videos(video_dir)
    if mp4:
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


# Convert RBG image to grayscale and normalize by data statistics
def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def hide_hud(img):
    img[84:] = 0
    return img


# Use to download colab parameter file
def download_colab_model(param_file):
    from google.colab import files
    files.download(param_file)


def save_as_onnx(torch_model, sample_input, model_path):
    torch.onnx.export(torch_model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      f=model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,
                      # the ONNX version to export the model to - see https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      )



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
            # self.gym_env.action_space.seed(seed)
        self.env, self.video_dir = self.wrap_env(self.gym_env)
        self.action_space = self.env.action_space
        self.img_stack = img_stack
        self.show_hud = show_hud

    def reset(self, raw_state=False):
        self.env, self.video_dir = self.wrap_env(self.gym_env)
        self.rewards = []
        self.disable_view_window()
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        if not self.show_hud:
            img_gray = hide_hud(img_gray)
        self.stack = [img_gray] * self.img_stack
        if raw_state:
            return np.array(self.stack), np.array(img_rgb)
        else:
            return np.array(self.stack)

    @staticmethod
    def disable_view_window():
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(self, *args, **kwargs):
            org_constructor(self, *args, **kwargs)
            self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor

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
