from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:

    batch_size = 32
    num_epochs = 55
    initial_learning_rate = 0.0001
    initial_weight_decay = 0.003
    "These parameters are really not great for training, but I have to move on to other things."

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "init_l": initial_learning_rate,
        "num_e": num_epochs,
        "weight_dec": initial_weight_decay
        
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(), lr=CONFIG.initial_learning_rate, weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
