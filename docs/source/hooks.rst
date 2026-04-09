Hooks
=====

Hooks allow custom logic to run at specific points during model initialisation
and inference. For instance, an :class:`~xpmir.context.InitializationHook` can
load external resources or modify model state before or after the standard
initialisation sequence.

.. currentmodule:: xpmir.context

.. autoxpmconfig:: Hook

.. autoxpmconfig:: InitializationHook
    :members: before, after
