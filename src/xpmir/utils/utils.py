import inspect
import logging
import os
from pathlib import Path
import re
from subprocess import run
import tempfile
from experimaestro import SubmitHook, Job, Launcher
from threading import Thread
from typing import (
    BinaryIO,
    Callable,
    Iterator,
    List,
    TextIO,
    TypeVar,
    Union,
    Iterable,
)
from .functools import cache

T = TypeVar("T")


class StreamGenerator(Thread):
    """Create a FIFO pipe (*nix only) that is fed by the provider generator"""

    def __init__(self, generator: Callable[[Union[TextIO, BinaryIO]], None], mode="wb"):
        super().__init__()
        tmpdir = tempfile.mkdtemp()
        self.mode = mode
        self.filepath = Path(os.path.join(tmpdir, "fifo.json"))
        os.mkfifo(self.filepath)
        self.generator = generator
        self.error = False

    def run(self):
        try:
            with self.filepath.open(self.mode) as out:
                try:
                    self.generator(out)
                except Exception:
                    # Just write something so the file is closed
                    if isinstance(out, TextIO):
                        out.write("")
                    else:
                        out.write(b"0")
                    raise
        except Exception:
            self.error = True
            raise

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.join()
        self.filepath.unlink()
        self.filepath.parent.rmdir()
        if self.error:
            raise AssertionError("Error with the generator")


class Handler:
    """Returns a result that depends on the type of the argument

    Example:
    ```
    handler = Handler()

    @handler()
    def trectopics(topics: TrecTopics):
        return ("-topicreader", "Trec", "-topics", topics.path)

    @handler()
    def tsvtopics(topics: ir_csv.Topics):
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


def foreach(iterator: Iterable[T], fn: Callable[[T], None]):
    for t in map(fn, iterator):
        pass


def batchiter(batchsize: int, iter: Iterator[T], keeppartial=True) -> Iterator[List[T]]:
    """Group items together to form of list of size `batchsize`"""
    samples = []
    for sample in iter:
        samples.append(sample)
        if len(samples) % batchsize == 0:
            yield samples
            samples = []

    # Yield last samples if keeppartial is true
    if keeppartial and len(samples) > 0:
        yield samples


def easylog():
    """
    Returns a logger with the caller's __name__
    """
    import inspect

    try:
        frame = inspect.stack()[1]  # caller
        module = inspect.getmodule(frame[0])
        return logging.getLogger(module.__name__)
    except IndexError:
        return logging.getLogger("UNKNOWN")


class EasyLogger:
    @property
    def logger(self):
        clsdict = self.__class__.__dict__

        logger = clsdict.get("__LOGGER__", None)
        if logger is None:
            logger = logging.getLogger(self.__class__.__qualname__)
            self.__class__.__LOGGER__ = logger

        return logger


@cache
def find_java_home(min_version: int = 6) -> str:
    """Find JAVA HOME"""
    paths = []

    # (1) Use environment variable
    if java_home := os.environ.get("FORCE_JAVA_HOME", None):
        return java_home

    if java_home := os.environ.get("JAVA_HOME", None):
        paths.append(Path(java_home) / "bin" / "java")

    # Try java
    paths.append("java")

    # (2) Use java -XshowSettings:properties
    for p in paths:
        try:
            p = run(
                [p, "-XshowSettings:properties", "-version"],
                check=True,
                capture_output=True,
            )

            if m := re.search(
                rb".*\n\s+java.version = (\d+)\.[\d\.]+(-\w+)?\n.*",
                p.stderr,
                re.MULTILINE,
            ):
                version = int(m[1].decode())
                if min_version <= version:
                    if m := re.search(
                        rb".*\n\s+java.home = (.*)\n.*", p.stderr, re.MULTILINE
                    ):
                        return m[1].decode()
                else:
                    logging.info(
                        "Java search (version >= %d): skipping %s", min_version, p
                    )

        except Exception:
            # silently ignore
            pass

    raise FileNotFoundError(f"Java (version >= {min_version}) not found")


class needs_java(SubmitHook):
    """Experimaestro hook that ensures that JAVA_HOME is set"""

    def __init__(self, version: int):
        self.version = version

    def spec(self):
        return self.version

    def process(self, job: Job, launcher: Launcher):
        job.environ["JAVA_HOME"] = find_java_home(self.version)


class Initializable:
    """Base class for all initializable (but just once)"""

    def initialize(self, *args, **kwargs):
        """Main initialization

        Calls :py:meth:`__initialize__` once (using :py:meth:`__initialize__`)
        """
        if not self._initialized:
            self._initialized = True
            self.__initialize__(*args, **kwargs)
        self._initialized = True

    def __init__(self):
        self._initialized = False

    def __initialize__(self, *args, **kwargs):
        """Initialize the object

        Parameters depend on the actual class
        """
        pass
