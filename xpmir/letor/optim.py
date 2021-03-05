from experimaestro import Config, Param


class Optimizer(Config):
    pass


class Adam(Optimizer):
    lr: Param[float] = 1e-3

    def __call__(self, parameters):
        from torch.optim import Adam

        return Adam(parameters, lr=self.lr)
