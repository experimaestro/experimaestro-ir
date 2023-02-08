from dataclasses import dataclass
from experimaestro import Config, Param
from experimaestro.compat import cached_property
import torch
from experimaestro.taskglobals import Env as TaskEnv
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile
from xpmir.utils.utils import easylog

logger = easylog()


@dataclass
class DeviceInformation:
    device: torch.device
    """The device"""

    main: bool
    """Flag for the main process (all other are slaves)"""


@dataclass
class DistributedDeviceInformation(DeviceInformation):
    rank: int
    """When using distributed processing, this is the rank of the process"""


class Device(Config):
    """Device to use, as well as specific option (e.g. parallelism)"""

    @cached_property
    def value(self):
        import torch

        return torch.device("cpu")

    def execute(self, callback):
        return callback(DeviceInformation(self.value, True))


def mp_launcher(rank, path, world_size, device, callback, taskenv):
    logger.warning("Launcher of rank %d [%s]", rank, path)
    TaskEnv._instance = taskenv
    taskenv.slave = rank == 0

    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=world_size
    )
    callback(DistributedDeviceInformation(device, rank == 0, rank))

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

    def execute(self, callback):
        # Setup distributed computation
        # Seehttps://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        n_gpus = torch.cuda.device_count()
        if n_gpus == 1 or not self.distributed:
            callback(DeviceInformation(self.value, True))
        else:
            with tempfile.NamedTemporaryFile() as temporary:
                logger.info("Setting up distributed CUDA computing (%d GPUs)", n_gpus)
                mp.spawn(
                    mp_launcher,
                    args=(
                        temporary.name,
                        n_gpus,
                        self.value,
                        callback,
                        TaskEnv.instance(),
                    ),
                    nprocs=n_gpus,
                    join=True,
                )


# Default device is the CPU
DEFAULT_DEVICE = Device()
