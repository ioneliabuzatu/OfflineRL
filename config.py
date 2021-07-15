import experiment_buddy

img_stack = 4
batchsize = 7496
show_hud = True
use_colab_autodownload = False
seed = 777
epochs = 500

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(host="", sweep_yaml="")