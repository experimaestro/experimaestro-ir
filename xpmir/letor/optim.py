from experimaestro import Config, Param


class Optimizer(Config):
    pass


class Adam(Optimizer):
    lr: Param[float] = 1e-3

    def __call__(self, parameters):
        from torch.optim import Adam

        return Adam(parameters, lr=self.lr)


class AdamW(Optimizer):
    """Adam optimizer that takes into account the regularization"""

    lr: Param[float] = 1e-3
    weight_decay: Param[float] = 1e-2

    def __call__(self, parameters):
        from torch.optim import AdamW

        return AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
