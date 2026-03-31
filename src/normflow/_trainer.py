# Created by Javad Komijani, 2025

"""This module contains high-level classes for training."""

import os
import csv
from typing import Dict, Callable, Protocol, Literal, Union
import time
import logging
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import pydantic


__all__ = ["Trainer"]


# =============================================================================
class Trainer:
    """
    High-level orchestration for model training.

    Overview:
        This class serves a role similar to a lightweight PyTorch Lightning
        Trainer. It manages the training loop, epoch tracking, optimizer and
        scheduler configuration, and distributed execution, while delegating
        model-specific logic to the model's `training_step` method.

    Key features:
        - Unified interface for single-device and multi-device (DDP) training
        - Centralized optimizer and scheduler configuration
        - Automatic device placement and process coordination
        - Checkpoint loading and saving (rank-safe in distributed mode)
        - Pluggable logging backend with sensible defaults
        - Deterministic and reproducible training via seed management

    Device selection:
        Single- or multi-GPU execution is determined automatically based on the
        runtime environment and launch method. When the script is launched via
        ``torchrun`` and the environment variable ``WORLD_SIZE`` is greater
        than 1, the Trainer uses Distributed Data Parallel (DDP) for multi-GPU
        training. When launched via standard ``python`` (or when
        ``WORLD_SIZE == 1``), training runs in a single process on one device
        (GPU if available, CPU otherwise).

    Training configuration:
        Training-related configuration (e.g. optimizer, scheduler, and
        hyperparameters) can be provided either at initialization time or
        when calling `run_training`. Configuration passed to `run_training`
        overrides values specified during initialization.

    Typical usage:
        >>> model = MyModel()
        >>> trainer = Trainer(model)
        >>> trainer.run_training(
        ...     n_epochs=10,
        ...     batch_size=8,
        ...     optimizer_class=torch.optim.AdamW,
        ...     hyperparam={"lr": 3e-4},
        ...     save_checkpoint_path="model.pt",
        ... )

    Responsibilities:
        - Maintain training state (epochs, optimizers, schedulers)
        - Configure and apply optimization strategies
        - Execute training loops (single-process or DDP)
        - Persist and restore model checkpoints
        - Coordinate logging and progress reporting
    """

    def __init__(
        self,
        model: torch.nn.Module,
        logger: Union["LoggerLike", None] = None,
        **training_config,
    ):
        """Initializes the trainer with the given model.

        Args:
            model (torch.nn.Module): The model to be trained.
            logger (LoggerLike or None): If None, uses the default `CSVLogger`.
            **training_config: Additional kweword arguments, including:
                optimizer_class: Callable = torch.optim.AdamW
                lr_scheduler_class: Callable | None = None
                path_gradient_autodiff: bool = True
                    A flag indicating whether gradient autodiff is used.
                alpha_tmax: int | None = None
                    If provided, activates `AlphaScheduler` for interpolation
                    between the prior and target distributions.
                hyperparam: Dict = {}

        Notes:
            Training configuration can be supplied either at initialization
            or when calling `run_training`. If the same configuration key is
            provided in both places, the value passed to `run_training` takes
            precedence.

            This design allows static (Lightning-style) configuration as well
            as dynamic, run-specific overrides.
        """
        self.model = model
        self.current_epoch = 0
        self.training_dataloader = None
        self.device_handler = DeviceHandler()
        self.logger = logger or CSVLogger()
        self.optimizer = None
        self.lr_scheduler = None
        self.alpha_scheduler = None
        self.config = TrainingConfiguration(**training_config)

    def configure_optimizers(self, **kwargs):
        """Configure the optimizers and logging."""

        if "log_name" in kwargs:
            self.logger.reset_name(kwargs["log_name"])
            kwargs.pop("log_name")

        self.config.update(**kwargs)

        # For initiating optimizer, get model parameters (grouped or flat)
        if '_groups' in self.model.net_.__dict__.keys():
            parameters = self.model.net_.grouped_parameters()
        else:
            parameters = self.model.net_.parameters()
        hyperparam = self.config.hyperparam
        self.optimizer = self.config.optimizer_class(parameters, **hyperparam)

        # Setup lr scheduler if the corresponding class is provided
        if self.config.lr_scheduler_class is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = self.config.lr_scheduler_class(self.optimizer)

        # Setup alpha scheduler if alpha_tmax is provided
        if self.config.alpha_tmax is None:
            self.alpha_scheduler = None
        else:
            self.alpha_scheduler = AlphaScheduler(self.config.alpha_tmax)

    def run_training(self, n_epochs: int, batch_size: int, **config):
        """Run the training workflow (distributed or non-distributed).

        This method performs the following steps:
        - Selects between single-process and Distributed Data Parallel (DDP)
          execution based on the current environment.
        - Loads a checkpoint if a `load_checkpoint_path` is provided.
        - Sets up the optimizer, scheduler, and other training components.
        - Runs the training loop for the specified number of epochs.
        - Saves a checkpoint after training if a `save_checkpoint_path` is
          provided (only on the main process when using distributed training).

        Args:
            n_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            **config: Optional training configuration. These parameters update
               or replace any configuration provided at initialization time
               (e.g. optimizer, scheduler, hyperparameters, logging name).
        """
        t_0 = time.time()
        if self.device_handler.world_size == 1:
            self._run_training(n_epochs, batch_size, **config)
        else:
            self._run_ddp_training(n_epochs, batch_size, **config)

        if self.is_main_process:
            logging.info("Training completed in %.1f sec.", time.time() - t_0)

    def _run_training(
        self,
        n_epochs: int,
        batch_size: int,
        load_checkpoint_path: str | None = None,
        save_checkpoint_path: str | None = None,
        **config
    ):
        """Run the training workflow.

        This method performs the following steps:
        - Loads a checkpoint if a `load_checkpoint_path` is provided.
        - Sets up the optimizer, scheduler, and other training components.
        - Runs the training loop for the specified number of epochs.
        - Saves a checkpoint after training if a `save_checkpoint_path` is
          provided (only on the main process when using distributed training).
        """
        self.load_checkpoint(load_checkpoint_path)
        self.device_handler.to_training_device(self.model.net_)
        self.device_handler.to_training_device(self.model.prior)
        self.configure_optimizers(**config)

        self.device_handler.print_device_info()
        print_model_info(self.model.net_)

        progress = tqdm(
            range(1 + self.current_epoch, 1 + self.current_epoch + n_epochs),
            disable=not self.is_main_process,
        )

        for self.current_epoch in progress:
            loss, logq, logp, ess = self.training_epoch(batch_size)
            self.logger.log_epoch(
                self.current_epoch,
                {'loss': loss, 'ess': ess, 'logp': logp.mean()}
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()

        self.save_checkpoint(save_checkpoint_path)

    def _run_ddp_training(
        self,
        n_epochs: int,
        batch_size: int,
        load_checkpoint_path: str | None = None,
        save_checkpoint_path: str | None = None,
        seeds_list: tuple | None = None,
        **config
    ):
        """Run distributed training using Distributed Data Parallel (DDP).

        Steps:
        1. Initialize the process group for distributed communication.
        2. Load the model if a valid path is provided.
        3. Wrap the model with DDP for multi-GPU training.
        4. Set random seeds for reproducibility.
        5. Execute the training routine.
        6. Save the model if a valid path is provided (only on main process).
        7. Synchronize all processes.
        8. Destroy the process group to free resources.
        """
        # Initialize distributed backend
        self.device_handler.init_process_group(backend="nccl")
        self.load_checkpoint(load_checkpoint_path)
        net_ = self.model.net_
        self.model.net_ = self.device_handler.model_ddp_wrapper(net_)
        self.device_handler.set_seed(seeds_list)

        if self.is_main_process:
            logging.info("Process group initialized & model wrapped with DDP.")

        try:
            # Execute training
            self._run_training(n_epochs, batch_size, **config)
            self.save_checkpoint(save_checkpoint_path)

            # Synchronize all processes after training
            torch.distributed.barrier()

        finally:
            # Without finally, cleanup runs only if everything succeeds
            self.device_handler.destroy_process_group()  # Cleanup
            if self.is_main_process:
                logging.info("Process group destroyed.")

    def training_epoch(self, batch_size: int):
        """
        Perform a single training step with a batch of size `batch_size`.

        Note that:
        - The alpha scheduler controls the interpolation between the action
          term and the prior during training.
        - If `path_gradient_autodiff` is enabled, the forward pass uses the
          `forward_with_path_gradient_ad` method of `Module_` to adjust
          autmatic differentiation.

        This method samples inputs from the prior distribution, computes
        transformed outputs, evaluates loss based on log-probabilities, and
        optimizes the model using backpropagation.
        """
        net_ = self.model.net_
        prior = self.model.prior
        action = self.model.action

        # Sample inputs from the prior
        x, logr = prior.sample_(batch_size)

        # Forward pass through the neural network
        y, logj = net_.forward(x)

        # Compute the log-probability of the transformed data `y`
        logq = logr - logj

        if self.config.path_gradient_autodiff:
            logq += adjustment_for_path_gradient_autodiff(y, net_, prior)

        # Compute target log-probability
        if self.alpha_scheduler is None:
            logp = - action(y)
        else:
            alpha = self.alpha_scheduler.alpha
            logp = - alpha * action(y) + (1 - alpha) * prior.log_prob(y)

        # Compute Loss
        loss = self.config.loss_fn(logq, logp)

        if torch.isnan(loss):
            raise RuntimeError("Stopping due to NaN loss")

        self.optimizer.zero_grad()  # clears old gradients from last steps
        loss.backward()

        if self.config.clip_grad_norm:
            clip_grad_norm_(net_.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss, logq, logp, Metrics.calc_ess(logq, logp)

    @property
    def is_main_process(self):
        """Return if main process."""
        return self.device_handler.is_main_process

    def save_checkpoint(self, fname: str):
        """Save the model state (on rank 0)."""
        if fname is None or not self.is_main_process:
            return
        if self.device_handler.world_size == 1:
            torch.save(self.model.net_.state_dict(), fname)
        else:
            torch.save(self.model.net_.module.state_dict(), fname)

    def load_checkpoint(self, fname: str, map_location=torch.device('cpu')):
        """Load a model checkpoint into the current instance of the model."""
        # When `torch.load()` is called, use the map_location argument to load
        # tensors onto the CPU first even when GPU exists, especially when
        # working with large models or limited GPU memory.
        if fname is None:
            return
        state = torch.load(fname, map_location=map_location, weights_only=True)
        self.model.net_.load_state_dict(state)


# =============================================================================
def adjustment_for_path_gradient_autodiff(y, net_, prior):
    """
    Compute the path gradient adjustment for statistical stability.

    In KL divergence minimization, the total derivative of the loss decomposes
    into a partial derivative with respect to parameters and the transformed
    variable `y`. The partial derivative contribution statistically vanishes,
    making it preferable to remove these terms using a reverse flow correction.
    The technique follows Vaitl et al., "Gradients should stay on Path: Better
    Estimators of the Reverse- and Forward KL Divergence for Normalizing Flows"
    [arXiv:2207.08219].

    Args:
        y (Tensor): Transformed variable obtained from the normalizing flow.
        net_ (Module): For applying the reverse flow on `y` to obtain `x`.
        prior: For computing the log probability of the `x`.

    Returns:
        Tensor: Adjustment to `logq` of `y`.
    """
    # Note that `y` is obtained through the forward transformation:
    # `y, logj = net_.forward(x)`

    # Compute the reverse transformation on detached `y` to calculate
    # the partial gradient w.r.t. only the parameters of the reverse path
    x, minus_logj = net_.reverse(y.detach())
    # Note that, basically, `minus_logj = -logj`

    # Adjust the log-Jacobian for statistical stability by removing the
    # parameter-related contributions to the gradient log-Jacobian
    d_logj = minus_logj - minus_logj.detach()
    # Note that, d_logj is zero, but its gradeint is the partial derivative of
    # logj w.r.t. the parameters

    # Compute the log-probability of `x` with adjustment for stability
    d_logr = - (prior.log_prob(x) - prior.log_prob(x).detach())

    d_logq = d_logr - d_logj

    return d_logq


# =============================================================================
class AlphaScheduler:
    """
    A class that introduces a parameter `alpha` and a scheduler to increment
    its value over time. The value of `alpha` starts at 0 and increases
    gradually towards 1 based on a given maximum number of steps (`t_max`).
    The value of `alpha` is updated in each step, and it never exceeds 1.
    """

    def __init__(self, t_max: int):
        """
        Initializes the AlphaScheduler with a maximum number of steps (`t_max`)
        over which `alpha` increases from 0 to 1.
        """
        self.alpha = 0
        self.t_max = t_max

    def step(self):
        """
        Increments the value of `alpha` by `1 / t_max`. The value of `alpha` is
        clamped to a maximum of 1.
        """
        self.alpha = min(1, self.alpha + 1 / self.t_max)


# =============================================================================
class Metrics:
    """Define various metrics"""

    @staticmethod
    def calc_kl_mean(logq, logp):
        """Return Kullback-Leibler divergence estimated from logq and logp."""
        return (logq - logp).mean()  # KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq, logp):
        """Return the variance of logq and logp difference."""
        return (logq - logp).var()

    @staticmethod
    def calc_corrcoef(logq, logp):
        """Return coreelation between logq and logp."""
        return torch.corrcoef(torch.stack([logq, logp]))[0, 1]

    @staticmethod
    def calc_direct_kl_mean(logq, logp):
        """Return direct KL divergence estimated from logq and logp."""
        logpq = logp - logq
        logz = torch.logsumexp(logpq, dim=0) - np.log(logp.shape[0])
        logpq = logpq - logz  # p is now normalized
        p_by_q = torch.exp(logpq)
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_minus_logz(logq, logp):
        """Return minus log of :math:`Z` estimated from logq and logp."""
        logz = torch.logsumexp(logp - logq, dim=0) - np.log(logp.shape[0])
        return -logz

    @staticmethod
    def calc_ess(logq, logp):
        """Rerturn effective sample size (ESS)."""
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) \
            - torch.logsumexp(-2*logqp, dim=0)
        ess = torch.exp(log_ess) / len(logqp)  # normalized
        return ess

    @staticmethod
    def calc_minus_logess(logq, logp):
        """Return logarith of inverse of effective sample size."""
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) \
            - torch.logsumexp(-2*logqp, dim=0)
        return - log_ess + np.log(len(logqp))  # normalized


# =============================================================================
class TrainingConfiguration(pydantic.BaseModel):
    """Training Configuration."""

    loss_fn: Callable = Metrics.calc_kl_mean
    optimizer_class: Callable = torch.optim.AdamW
    lr_scheduler_class: Callable | None = None
    alpha_tmax: int | None = None
    path_gradient_autodiff: bool = True
    hyperparam: Dict = {}
    clip_grad_norm: bool = False

    def update(self, **kwargs):
        """Update the attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
class DDP(torch.nn.parallel.DistributedDataParallel):
    """Wrapes a Module such that its attributes become accessible with DDP."""
    # After wrapping a Module with DistributedDataParallel, the attributes
    # of the module (e.g. custom methods) became inaccessible. To access them,
    # a workaround is to use a subclass of DistributedDataParallel as here.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DeviceHandler:
    """A handler for managing device setup and distributed training.

    Key Features:
        - Supports CPU, single GPU, and multi-GPU distributed training.
        - Automatically initializes and destroys process groups when needed.
        - Moves models to the appropriate device.
        - Wraps models with DistributedDataParallel for multi-GPU training.
        - Provides utilities like main-process detection & reproducible seeds.
    """
    def __init__(
        self,
        training_device: Literal["gpu", "cpu", "auto"] = "auto"
    ):
        """Initialize for 1 rank. If needed will be changed later."""
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size == 1:
            self.rank = 0  # The global rank of the current process
            self.local_rank = 0  # The rank within the local node (GPU)
            flag = torch.cuda.is_available() and training_device != "cpu"
            self.training_device = "cuda" if flag else "cpu"
        else:
            # to be de determined later in self.init_process_group()
            self.rank = None  # The global rank of the current process
            self.local_rank = None  # The rank within the local node (GPU)
            self.training_device = None

    def set_seed(self, seeds_list=None):
        """Sets the random seeds across distributed processes.

        Args:
            seeds_list (Tuple): A list of seeds, one for each process.
            If None, no seed is set.
        """
        if seeds_list is not None:
            seed = seeds_list[self.rank]
            torch.manual_seed(seed)

    def all_gather_into_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Gather all tensors if `world_size > 1`."""
        if self.world_size == 1:
            return x

        out_shape = list(x.shape)
        out_shape[0] *= self.world_size
        out = torch.zeros(*out_shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out

    @property
    def is_main_process(self) -> bool:
        """Return True if the current process is the main process (rank 0)."""
        return self.rank == 0

    @property
    def is_main_worker(self):
        """Return True if the current worker is the main one (local rank 0)."""
        return self.local_rank == 0

    def init_process_group(self, backend="nccl"):
        """Initializes the distributed process group for multi-GPU training.

        Args:
            backend (str): The backend for communication (e.g. 'nccl' for GPU).

        Note:
            Sets the world size, rank, and local rank of the current process.
        """
        assert torch.cuda.is_available()

        # Initialize distributed backend
        dist.init_process_group(backend=backend)

        self.world_size = dist.get_world_size()

        gpus_per_node = torch.cuda.device_count()

        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank() % gpus_per_node
        self.training_device = torch.device(f"cuda:{self.local_rank}")

    def destroy_process_group(self):
        """Destroys the distributed process group and cleans up the resources.

        After calling this, no further distributed operations can be performed
        until re-initialization.
        """
        dist.destroy_process_group()

    def model_ddp_wrapper(self, model):
        """
        Moves the model to the device specified with the local rank, and
        wraps the model in DistributedDataParallel (DDP).
        """
        device = self.training_device
        model.to(device=device, dtype=None)
        model = DDP(model, device_ids=[device], output_device=device)
        return model

    def to_training_device(self, x):
        """Moves all args to self.training_device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.training_device)
        if isinstance(x, list):
            return [z.to(self.training_device) for z in x]
        if isinstance(x, tuple):
            return tuple(z.to(self.training_device) for z in x)
        if isinstance(x, torch.nn.Module):
            return x.to(self.training_device, dtype=None)
        if hasattr(x, "to"):
            return x.to(self.training_device)
        raise ValueError("Only Tensor, list, tuple are supported.")

    def print_device_info(self):
        """Print device info."""
        if self.world_size > 1:
            torch.distributed.barrier()
        logging.info("Utilized training device: %s", self.training_device)
        if self.world_size > 1:
            torch.distributed.barrier()


# =============================================================================
class CSVLogger:
    """Minimal CSV logger with Lightning-style versioning.

    If name is not None, creates:
        logs/name/version_{?}/metrics.csv

    Usage:
        logger = CSVLogger("logs", "experiment")
        logger.log_epoch(epoch, {"loss": loss})
    """
    def __init__(
        self,
        root: str | None = None,
        name: str | None = None,
        every_n_epochs: int = 1
    ):
        self.root = root or ''
        self.name = name
        self.every_n_epochs = every_n_epochs

        self.buffer = []  # Holds only unwritten entries
        self.path = None
        self.file = None
        self._file_initialized = False
        self._header_written = False

    def _dist_ready(self):
        return dist.is_available() and dist.is_initialized()

    def _is_rank0(self):
        return (not self._dist_ready()) or dist.get_rank() == 0

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Average a tensor across all ranks. Input must be a tensor."""
        if not self._dist_ready():
            return x

        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / dist.get_world_size()

    def _next_version(self, base):
        versions = [
            int(d.split("_")[-1])
            for d in os.listdir(base)
            if d.startswith("version_") and d.split("_")[-1].isdigit()
        ]
        return max(versions, default=-1) + 1

    def _setup_file(self):
        if self._file_initialized or self.name is None:
            return

        if not self._is_rank0():
            return

        base = os.path.join(self.root, self.name)
        os.makedirs(base, exist_ok=True)

        version = self._next_version(base)
        self.path = os.path.join(base, f"version_{version}")
        os.makedirs(self.path, exist_ok=True)

        self.file = os.path.join(self.path, "metrics.csv")

        # create file if missing (but don't truncate)
        if not os.path.exists(self.file):
            with open(self.file, "a", encoding="utf-8"):
                pass  # just create the file

        self._file_initialized = True
        self._header_written = os.path.getsize(self.file) > 0

    def reset_name(self, name):
        """Reset 'self.name' if different from 'name'."""
        if self.name == name:
            return
        self.name = name
        self.path = None
        self.file = None
        self._file_initialized = False
        self._header_written = False

    def log_epoch(self, epoch: int, metrics: Dict):
        """Log metrics for given epoch."""

        def reduce_to_float(x):
            return self._reduce(x).item() if isinstance(x, torch.Tensor) else x

        metrics = {k: reduce_to_float(v) for k, v in metrics.items()}
        entry = {"epoch": epoch, **metrics}

        if self.buffer and self.buffer[-1]["epoch"] == epoch:
            self.buffer[-1].update(entry)
        else:
            self.buffer.append(entry)

        if self.name is not None and epoch % self.every_n_epochs == 0:
            self.flush()

    def flush(self):
        """Dump buffer safely to disk (rank0 only) and clears it."""
        if not self.buffer:
            return

        if not self._is_rank0():
            self.buffer.clear()
            return

        if not self._file_initialized:
            self._setup_file()

        fields = self.buffer[0].keys()

        with open(self.file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if not self._header_written:
                writer.writeheader()
                self._header_written = True

            for row in self.buffer:
                writer.writerow(row)

        self.buffer.clear()

    def load_numpy(self):
        """Return all logged metrics as a dict of stacked NumPy arrays.

        If the CSV file exists, load from it. Otherwise, return the in-memory
        buffer. Non-numeric entries are preserved as object arrays.
        """
        # Decide source: file or buffer
        if self.file is not None and os.path.exists(self.file):
            # Load from CSV file
            with open(self.file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                data = {k: [] for k in columns}
                for row in reader:
                    for k in columns:
                        try:
                            data[k].append(float(row[k]))
                        except ValueError:
                            data[k].append(row[k])
        else:
            # Load from in-memory buffer
            if not self.buffer:
                return {}
            columns = self.buffer[0].keys()
            data = {k: [] for k in columns}
            for row in self.buffer:
                for k in columns:
                    try:
                        data[k].append(float(row[k]))
                    except ValueError:
                        data[k].append(row[k])

        # Convert lists to numpy arrays
        for k in data:
            try:
                data[k] = np.array(data[k], dtype=float)
            except Exception:
                data[k] = np.array(data[k], dtype=object)

        return data


class LoggerLike(Protocol):
    """Any logger-like object that can accept log_epoch."""
    def log_epoch(self, epoch: int, metrics: Dict) -> None:
        """Log metrics for given epoch."""


# =============================================================================
def print_model_info(model):
    """Print Model summary."""

    if int(os.environ.get("RANK", 0)) != 0:
        return  # Only main worker prints full startup info

    # Model summary
    if dist.is_available() and dist.is_initialized():
        model = model.module  # because model is wrapped
    parameters = list(model.parameters())
    total_params = sum(p.numel() for p in parameters)
    trainable_params = sum(p.numel() for p in parameters if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    def print_format(i, a, b, c, d):
        print(f"{i} | {a[:10]:<10} | {b[:20]:<20} | {c:<6} | {d}")

    print("\n  | Name       | Type                 | Params | Mode ")
    print("-" * 54)
    for i, (name, module) in enumerate(model.named_children()):
        params = sum(p.numel() for p in module.parameters())
        mode = "train" if module.training else "eval"
        print_format(i, name, type(module).__name__, params, mode)
    print("-" * 54)
    print(f"{trainable_params}\tTrainable params")
    print(f"{non_trainable_params}\tNon-trainable params")
    print(f"{total_params}\tTotal params\n")
    print(f"Default dtype: {torch.ones(1).dtype}\n")
