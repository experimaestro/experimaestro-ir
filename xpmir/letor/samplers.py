from experimaestro import config


@config()
class Sampler:
    """"Abtract data sampler"""

    pass


@config()
class ModelBasedSampler(Sampler):
    pass
