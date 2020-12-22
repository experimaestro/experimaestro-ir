from experimaestro import config, param


@config()
class Optimizer:
    pass


@param("lr", default=1e-3)
@config()
class Adam(Optimizer):
    def __call__(self, parameters):
        from torch.optim import Adam

        return Adam(parameters, lr=self.lr)
