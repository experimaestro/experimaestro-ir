import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from experimaestro import Config, Param
from experimaestro.compat import cached_property
import torch
from experimaestro.taskglobals import Env as TaskEnv
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile
from xpmir.context import Context
from xpmir.utils.utils import easylog

logger = easylog()


@dataclass
class DeviceInformation:
    device: torch.device
    """The device"""

    main: bool
    """Flag for the main process (all other are slaves)"""

    count: int = 1
    """Number of processes"""

    rank: int = 0
    """When using distributed processing, this is the rank of the process"""


class ComputationContext(Context):
    device_information: DeviceInformation


@dataclass
class DistributedDeviceInformation(DeviceInformation):
    pass


class Device(Config):
    """Device to use, as well as specific option (e.g. parallelism)"""

    @cached_property
    def value(self):
        import torch

        return torch.device("cpu")

    n_processes = 1
    """Number of processes"""

    def execute(self, callback, *args, **kwargs):
        callback(DeviceInformation(self.value, True), *args, **kwargs)


def mp_launcher(rank, path, world_size, callback, taskenv, args, kwargs):
    logger.info("Started process for rank %d [%s]", rank, path)
    TaskEnv._instance = taskenv
    taskenv.slave = rank == 0

    logger.info("Initializing process group [%d]", rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=world_size
    )

    logger.info("Calling callback [%d]", rank)
    device = torch.device(f"cuda:{rank}")
    callback(
        DistributedDeviceInformation(
            device=device, main=rank == 0, rank=rank, count=world_size
        ),
        *args,
        **kwargs,
    )

    # Cleanup
    dist.destroy_process_group()


class CudaDevice(Device):
    """CUDA device"""

    gpu_determ: Param[bool] = False
    """Sets the deterministic"""

    cpu_fallback: Param[bool] = False
    """Fallback to CPU if no GPU is available"""

    distributed: Param[bool] = False
    """Flag for using DistributedDataParallel

    When the number of GPUs is greater than one, use
    torch.nn.parallel.DistributedDataParallel when `distributed` is `True` and
    the number of GPUs greater than 1. When False, use `torch.nn.DataParallel`
    """

    @cached_property
    def value(self):
        """Called by experimaestro to substitute object at run time"""
        if not torch.cuda.is_available():
            if not self.cpu_fallback:
                # Not accepting fallbacks
                raise AssertionError("No GPU available")
            logger.error("No GPU available. Falling back on CPU.")
            return torch.device("cpu")

        # Set the deterministic flag
        torch.backends.cudnn.deterministic = self.gpu_determ
        if self.gpu_determ:
            logger.debug("using GPU (deterministic)")
        else:
            logger.debug("using GPU (non-deterministic)")

        return torch.device("cuda")

    @cached_property
    def n_processes(self):
        """Number of processes"""
        if self.distributed:
            return torch.cuda.device_count()
        return 1

    def execute(self, callback, *args, **kwargs):
        # Setup distributed computation
        # Seehttps://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        n_gpus = torch.cuda.device_count()
        assert torch.cuda.device_count() > 0
        if n_gpus == 1 or not self.distributed:
            callback(DeviceInformation(self.value, True), *args, **kwargs)
        else:
            if sys.version_info.major == 3 and sys.version_info.minor < 10:
                tmp_directory = tempfile.TemporaryDirectory()
            else:
                tmp_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

            with tmp_directory as directory:
                logger.info("Setting up distributed CUDA computing (%d GPUs)", n_gpus)
                return mp.start_processes(
                    mp_launcher,
                    args=(
                        str((Path(directory) / "link").absolute()),
                        n_gpus,
                        callback,
                        TaskEnv.instance(),
                        args,
                        kwargs,
                    ),
                    nprocs=n_gpus,
                    join=True,
                    start_method=mp.get_start_method(),
                )


class BestDevice(Device):
    """Try to use a GPU device if it exists, fallbacks to CPU otherwise
    
    To be used when debugging"""

    @cached_property
    def value(self):
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            logging.info(f"Using CPU: {torch.device('cpu')}")
        return device


# Default device is the CPU
DEFAULT_DEVICE = Device()
