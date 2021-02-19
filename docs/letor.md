---
title: Learning to rank
---

# Learning to rank

## Scores

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

::xpm::xpmir.rankers.Scorer


## Trainers

Trainers are responsible for defining the loss (given a learnable scorer)


::xpm::xpmir.letor.trainers.Trainer

## Sampler

How to sample learning batches.

::xpm::xpmir.letor.samplers.Sampler
