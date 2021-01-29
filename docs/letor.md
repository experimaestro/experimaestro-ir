---
title: Learning to rank
---

## Scorers

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

::xpm::xpmir.rankers.Scorer


## Trainers

Trainers are responsible for defining the loss (given a learnable scorer)


::xpm::xpmir.rankers.Trainer

## Sampler

How to sample learning batches.

::xpm::xpmir.rankers.Sampler



## Tasks

::xpm::xpmir.letor.learner.Learner
