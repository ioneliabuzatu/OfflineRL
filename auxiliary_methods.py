import shutil
import zipfile

seed = 777
import random

if seed: random.seed(seed)
import numpy as np

if seed: np.random.seed(seed)
import torch

if seed: torch.manual_seed(seed)

# Imports
import os
import time
# PyTorch imports
# Onnx model-export imports

# Auxiliary Python imports
import glob
import io
import base64
from time import time, strftime

# Environment import and set logger level to display error only
from gym import logger as gymlogger

gymlogger.set_level(40)  # error only

# Plotting and notebook imports
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()


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
def download_and_extract_data():
    os.system("wget --no-check-certificate 'https://cloud.ml.jku.at/s/CdYdidkkBpFgcED/download' -O train_mixed.zip")
    # select as a data root the mixed demonstratoins directory
    data_root = 'data_mixed'
    with zipfile.ZipFile('train_mixed.zip', 'r') as zip_ref:
        os.makedirs(data_root, exist_ok=True)
        zip_ref.extractall(data_root)


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
