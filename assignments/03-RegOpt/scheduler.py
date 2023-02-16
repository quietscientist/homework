from typing import List
from torch.optim.lr_scheduler import _LRScheduler
from config import CONFIG # do I even need this? 

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, init_l=0.01, num_e=2, weight_dec=0):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.optimizer = optimizer
        self.init_l = CONFIG.lrs_kwargs["init_l"]
        self.num_e = CONFIG.lrs_kwargs["num_e"]
        self.weight_dec = CONFIG.lrs_kwargs["weight_dec"]

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)
        """
        Exponential decay of the learning rate.
        """           
        inital_lr = self.init_l
        num_epochs = self.num_e     
        schedule = [inital_lr]
        
        for i in range(1,num_epochs):

            schedule.append(self.init_l*(1-self.weight_dec)**(i))
        
        return schedule
