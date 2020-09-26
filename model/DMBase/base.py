import torch
import torch.nn as nn


class DMBase(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(DMBase, self).__init__()
        self.config = config
        self.device = torch.device(config.device_str)
