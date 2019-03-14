from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWithRestartScheduler(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cyclic cosine annealing schedule
    :math:`T` is total number of steps in the training (:math:`total_epochs * steps_per_epoch`),
    :math:`M` is the number of cycles wanted in the training,
    :math:`lr_max` is the initial cycle lr and :math:`lr_min` the final cycle lr.
    
    At next cycles, new lr is computed with method compute_lr_min_max from from lr_min, lr_max, last_epoch and cur_cycle that you can override if you want dynamic lr_min/lr_max depending on epoch and cycle.

    .. math::

        \alpha(t) = {\lr_min} + \frac ({\lr_max} - {\lr_min}) {2} (cos (\pi * \frac {mod(t - 1, \lceil\frac {T} {M}\rceil} {\frac {T} {M}}) + 1)

    When last_epoch=-1, sets initial lr as lr_max.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_min (float): the minimum learning rate (at end of cycle).
        lr_max (float): the maximum learning rate (at start of cycle).
        total_epochs (int): The total number of epochs in training
        steps_per_epoch (int): The total number of steps per epoch (len(dataloader)).
        nb_cycles (int): The number of cycles in the cyclic annealing.
        last_epoch (int): The index of last step (it is not an epoch in . Default: -1.
        new_lr_min_max (lambda): a lambda computing new lr_min and lr_max from lr_min, lr_max, last_epoch and cur_cycle, by default using same lr_min and lr_max


    .. Inspired by
        - Snapshot Ensembles: Train 1, get M for free https://arxiv.org/abs/1704.00109
        - SGDR : Stochastic Gradient Descent with Warm Restarts https://arxiv.org/abs/1608.03983
        - Pytorch CosineAnnealing LR Scheduler https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """

    def __init__(self, optimizer,
                 lr_min, lr_max,
                 total_epochs, steps_per_epoch, nb_cycles,                
                 last_epoch=-1):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.total_epochs = total_epochs
        self.nb_cycles = nb_cycles
        self.steps_per_epoch = steps_per_epoch
        
        self.steps_per_cycle = math.ceil((self.total_epochs * self.steps_per_epoch) / self.nb_cycles)
        # current cycle
        self.cur_cycle = 1
        self.cur_lr_min, self.cur_lr_max = self.compute_lr_min_max(self.lr_min, self.lr_max, last_epoch, self.cur_cycle)
        self.cur_lr = self.cur_lr_max
        
        super(CosineAnnealingWithRestartScheduler, self).__init__(optimizer, last_epoch)
    
    def compute_lr_min_max(self, lr_min, lr_max, last_epoch, cur_cycle):
        return (lr_min, lr_max)
    
    def get_lr(self):
        # in snapshot ensemble, last_epoch is not about epochs but batch step as we update LR at each iteration
        cur_step = self.last_epoch
        
        # update cycle if reached steps_per_cycle steps in current cycle
        if cur_step > self.cur_cycle * self.steps_per_cycle:
            self.cur_cycle += 1
            self.cur_lr_min, self.cur_lr_max = self.compute_lr_min_max(self.lr_min, self.lr_max, self.last_epoch, self.cur_cycle)
            
        self.cur_lr = self.cur_lr_min + 0.5 * (self.cur_lr_max - self.cur_lr_min) * (1 + math.cos(math.pi * (cur_step % self.steps_per_cycle) / self.steps_per_cycle))
        
        # not using base_lr here
        return [
            self.cur_lr
            for base_lr in self.base_lrs
        ]
