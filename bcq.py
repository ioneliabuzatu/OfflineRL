import math

import torch.nn as nn
from torch.nn.functional import log_softmax

activation_mapping = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "identity": nn.Identity()}


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
            self._weights_init(self.cnn_1)
            self._weights_init(self.cnn_2)
            self._weights_init(self.cnn_3)
            self._weights_init(self.q_vals_l1)
            self._weights_init(self.q_vals_l2)
            self._weights_init(self.policy_output_l1)
            self._weights_init(self.policy_output_l2)

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

        return self.q_vals_l2(q_vals_l1),  logits
