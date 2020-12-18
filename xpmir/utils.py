from logging import Logger
import inspect


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

    def __call__(self):
        def annotate(method):
            spec = inspect.getfullargspec(method)
            assert len(spec.args) == 1 and spec.varargs is None

            self.handlers[spec.annotations[spec.args[0]]] = method

        return annotate

    def __getitem__(self, key):
        return self.handlers[key.__class__](key)


def logger():
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
