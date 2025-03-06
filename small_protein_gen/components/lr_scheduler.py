from torch.optim.lr_scheduler import SequentialLR


class LazySequentialLR(SequentialLR):
    def __init__(self, optimizer, schedulers, *args, **kwargs) -> None:
        resolved_schedulers = [scheduler(optimizer) for scheduler in schedulers]
        super().__init__(optimizer, resolved_schedulers, *args, **kwargs)
