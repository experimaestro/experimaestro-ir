from logging import Logger
import inspect
import logging


class Handler:
    """Returns a result that depends on the type of the argument

    Example:
    ```
    handler = Handler()

    @handler()
    def trectopics(topics: TrecAdhocTopics):
        return ("-topicreader", "Trec", "-topics", topics.path)

    @handler()
    def tsvtopics(topics: ir_csv.AdhocTopics):
        return ("-topicreader", "TsvInt", "-topics", topics.path)

    command.extend(handler[topics])

    ```
    """

    def __init__(self):
        self.handlers = {}
        self.defaulthandler = None

    def default(self):
        assert self.defaulthandler is None

        def annotate(method):
            self.defaulthandler = method
            return method

        return annotate

    def __call__(self):
        def annotate(method):
            spec = inspect.getfullargspec(method)
            assert len(spec.args) == 1 and spec.varargs is None

            self.handlers[spec.annotations[spec.args[0]]] = method

        return annotate

    def __getitem__(self, key):
        handler = self.handlers.get(key.__class__, None)
        if handler is None:
            if self.default is None:
                raise RuntimeError(
                    f"No handler for {key.__class__} and no default handler"
                )
            handler = self.defaulthandler

        return handler(key)


def easylog():
    """
    Returns a logger with the caller's __name__
    """
    import inspect

    try:
        frame = inspect.stack()[1]  # caller
        module = inspect.getmodule(frame[0])
        return Logger(module.__name__)
    except IndexError:
        return Logger("UNKNOWN")


class EasyLogger:
    @property
    def logger(self):
        clsdict = self.__class__.__dict__

        logger = clsdict.get("__LOGGER__", None)
        if logger is None:
            logger = logging.getLogger(self.__class__.__qualname__)
            self.__class__.__LOGGER__ = logger

        return logger
