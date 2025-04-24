# Copyright (c) 2021-2024 Javad Komijani

"""
This module contains high-level classes for normalizing flow techniques,
with the central `Model` class integrating essential components such as priors,
networks, and actions. It provides utilities for training and sampling,
along with support for MCMC sampling and device management.
"""

import time
import logging
import torch
import numpy as np

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import fmt_val_err
from .device import ModelDeviceHandler


# =============================================================================
class Model:
    """
    The central high-level class of the package, which integrates instances of
    essential classes (`prior`, `net_`, and `action`) to provide utilities for
    training and sampling. This class interfaces with various core components
    to facilitate training, posterior inference, MCMC sampling, and device
    management.

    Parameters
    ----------
    prior : instance of a `Prior` class
        An instance of a Prior class (e.g., `NormalPrior`) representing the
        model's prior distribution.

    net_ : instance of a `Module_` class
        A model component responsible for the transformations required in the
        model. The trailing underscore indicates that the associated forward
        method computes and returns the Jacobian of the transformation, which
        is crucial in the method of normalizing flows.

    action : instance of an `Action` class
        Defines the model's action, which specified the target distribution
        during training.

    load_checkpoint_path : str or None, optional
        If a string is provided, it is passed to the `load_checkpoint` method
        for loading a checkpoint.

    Attributes
    ----------
    trainer : Trainer
        An instance of the Trainer class, responsible for training the model.
        For training one can call `trainer`. Note that `trainer.__call__` is
        aliased to `train` as well as to `fit` for flexibility in usage.
        Moreover, `trainer.execute_ddp_training` is a method that cab be used
        for parallel training, which is also aliased to `execute_ddp_training`.

    posterior : Posterior
        An instance of the Posterior class, which manages posterior inference
        on the model parameters.

    mcmc : MCMCSampler
        An instance of the MCMCSampler class, enabling MCMC sampling for
        posterior distributions.

    blocked_mcmc : BlockedMCMCSampler
        An instance of the BlockedMCMCSampler class, providing blockwise
        MCMC sampling for improved sampling efficiency.

    device_handler : ModelDeviceHandler
        Manages the device (CPU/GPU) for model training and inference, ensuring
        seamless operation across hardware setups.
    """

    def __init__(self, *, prior, net_, action, load_checkpoint_path=None):

        # Main components of the model
        self.net_ = net_
        self.prior = prior
        self.action = action

        # Components for training
        self.trainer = Trainer(self)
        self.train = self.trainer.execute
        self.fit = self.train  # Alias for `train`
        self.execute_ddp_training = self.trainer.execute_ddp_training

        # Components for sampling
        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)

        # Components for device handling and loading saved models
        self.device_handler = ModelDeviceHandler(self)

        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)

    def state_dict(self):
        """Returns a dictionary containing the state of the model."""

        return {
            'net_state_dict': self.net_.state_dict(),
            'prior_state_dict': {},  # self.prior.state_dict(),
            'action_state_dict': {},  # self.action.state_dict(),
            'train_state_dict': self.trainer.state_dict()
        }

    def save_checkpoint(self, path):
        """Saves the state of the model in the given path (file)."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(
        self,
        path: str,
        parameters_only: bool = True,
        map_location=torch.device('cpu')
    ):
        """
        Load a model checkpoint into the current instance.

        This method restores the model's state from a checkpoint file. It
        supports loading only the parameters of the network or both parameters
        and additional states (e.g., optimizer or training-related states)
        depending on the `parameters_only` flag.

        Parameters
        ----------
        path : str
            Path to the checkpoint file to load. The checkpoint should be saved
            in a format compatible with PyTorch's `torch.save()` and contain
            specific state dictionaries (e.g., 'net_state_dict').

        parameters_only : bool, optional
            If `True`, only the model's parameters (network weights) are
            loaded. If `False`, additional states such as prior, action, and
            training states are also restored. Defaults to `True`.

        map_location : torch.device, optional
            Specifies how to map storage locations when loading the checkpoint.
            Defaults to `torch.device('cpu')`.

        Raises:
            FileNotFoundError:
                If the file specified by `path` does not exist.
            KeyError:
                If the checkpoint file does not contain the required state
                keys.

        Notes:
            - When `torch.load()` is called on a file containing GPU tensors,
              those tensors are loaded directly to the GPU by default. To
              avoid a surge in GPU memory usage, use the `map_location`
              argument to load tensors onto the CPU first, especially when
              working with large models or limited GPU memory.
        """
        state = torch.load(path, map_location=map_location, weights_only=True)

        self.net_.load_state_dict(state['net_state_dict'])

        if not parameters_only:
            self.prior.load_state_dict(state['prior_state_dict'])
            self.action.load_state_dict(state['action_state_dict'])
            self.trainer.load_state_dict(state['train_state_dict'])


# =============================================================================
class Posterior:
    """
    Creates samples directly from a trained probabilistic model.

    The `Posterior` class generates samples from a specified model without
    using an accept-reject step, making it suitable for tasks that require
    quick, direct sampling.

    Parameters
    ----------
    model : Model
        A trained model to sample from.

    Methods
    -------
    sample(batch_size=1, **kwargs)
        Returns a specified number of samples from the model.

    sample_(batch_size=1, preprocess_func=None)
        Returns samples and their log probabilities, with an optional
        preprocessing function.

    sample__(batch_size=1, **kwargs)
        Similar to `sample_`, but also returns the log probability of the
        target distribution.

    log_prob(y)
        Computes the log probability of given samples.
    """

    def __init__(self, model):
        self._model = model

    def sample(self, batch_size=1, **kwargs):
        """
        Draws samples from the model.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        Returns
        -------
        Tensor
            Generated samples.
        """
        return self.sample_(batch_size=batch_size, **kwargs)[0]

    def sample_(self, batch_size=1, preprocess_func=None):
        """
        Draws samples and their log probabilities from the model.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        preprocess_func : function or None, optional
            A function to adjust the prior samples if needed. It should take
            samples and log probabilities as input and return modified values.

        Returns
        -------
        tuple
            - `y`: Generated samples.
            - `logq`: Log probabilities of the samples.
        """
        x, logr = self._model.prior.sample_(batch_size)

        if preprocess_func is not None:
            x, logr = preprocess_func(x, logr)

        y, logj = self._model.net_(x)
        logq = logr - logj
        return y, logq

    def sample__(self, batch_size=1, **kwargs):
        """
        Similar to `sample_`, but also returns the log probability of the
        target distribution from `model.action`.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        Returns
        -------
        tuple
            - `y`: Generated samples.
            - `logq`: Log probabilities of the samples.
            - `logp`: Log probabilities from the target distribution.
        """
        y, logq = self.sample_(batch_size=batch_size, **kwargs)
        logp = -self._model.action(y)  # logp is log(p_{non-normalized})
        return y, logq, logp

    def log_prob(self, y):
        """
        Computes the log probability of the provided samples.

        Parameters
        ----------
        y : torch.Tensor
            Samples for which to calculate the log probability.

        Returns
        -------
        Tensor
            Log probabilities of the samples.
        """
        x, minus_logj = self._model.net_.reverse(y)
        logr = self._model.prior.log_prob(x)
        logq = logr + minus_logj
        return logq


# =============================================================================
class Trainer:
    """
    A class responsible for training a model, handling optimization, loss
    computation, and scheduling tasks.

    This class provides functionality for setting up and running the training
    loop with various configurable parameters such as optimizer, loss function,
    and learning rate scheduler. The `Trainer` tracks key statistics such as
    loss and supports the use of checkpoints to monitor training progress.

    Attributes
    ----------
    path_gradient_autodiff : bool
        A flag indicating whether gradient autodiff is used.

    optimizer_class : torch.optim.Optimizer
        The class for the optimizer to be used during training (default:
        torch.optim.AdamW).

    optimizer : torch.optim.Optimizer, optional
        The optimizer object used in training. Set automatically using
        `optimizer_class`.

    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The learning rate scheduler used during training.

    alpha_scheduler : AlphaScheduler, optional
        A scheduler for controlling interpolation between prior and target
        distributions.

    train_history : dict
        A dictionary tracking training statistics such as epoch number, loss,
        ESS (Effective Sample Size), and log probabilities.

    hyperparam : dict
        A dictionary for storing hyperparameters like learning rate and decay
        weights.

    checkpoint_dict : dict
        A dictionary used to configure printing and checkpointing behavior
        during training.

    loss_func : function
        The loss function used during training. By default, it is set to
        calculate KL divergence.
    """

    path_gradient_autodiff = True
    optimizer_class = torch.optim.AdamW
    optimizer = None
    scheduler = None
    alpha_scheduler = None

    def __init__(self, model: Model):
        """
        Initializes the trainer with the given model.

        Parameters
        ----------
        model : Model
            The model to be trained.
        """
        self._model = model

        # Initialize training history tracking
        self.train_history = \
            {'epoch': 0, 'loss': [], 'ess': [], 'logp': [], 'logqp': []}

        # Default hyperparameters
        self.hyperparam = {'fused': False}

        # Checkpoint configuration
        self.checkpoint_dict = {'print_every': None, 'print_bsize': None}

        # Default loss function
        self.loss_func = self.calc_kl_mean

    def setup_optimizer_and_components(
        self,
        optimizer_class=None,
        scheduler=None,
        loss_func=None,
        alpha_tmax=None,
        hyperparam=None,
        checkpoint_dict=None
    ):
        """
        Executes the training loop with the specified configuration.

        Parameters
        ----------
        optimizer_class : type, optional
            Optimizer class to use. If provided, replaces the current optimizer
            class and reinitializes the optimizer. By default, it uses AdamW.

        scheduler : type, optional
            Learning rate scheduler constructor. If provided, a scheduler will
            be created using the initialized optimizer.

        loss_func : function, optional
            Loss function to use. Defaults to KL divergence if not provided.

        alpha_tmax : int, optional
            Maximum steps for alpha interpolation between the prior and target
            distributions over the course of training. If provided, activates
            an `AlphaScheduler`.

        hyperparam : dict, optional
            Dictionary of optimizer hyperparameters, like `lr`. If provided,
            updates internal hyperparams and reinitializes the optimizer with
            the new values.

        checkpoint_dict : dict, optional
            Dictionary to control training progress output and checkpointing.

        Notes
        -----
        The optimizer is (re)initialized under the following conditions:
        - If an `optimizer_class` is explicitly provided, it replaces the
          default and reinitializes the optimizer.
        - If `hyperparam` is provided, even if the optimizer already exists, it
          is reinitialized with the updated values.
        - If neither is provided, and the optimizer has not yet been
          initialized.
        """
        # Flag to decide whether the optimizer needs to be (re)initialized
        if self.optimizer is None:
            # If no optimizer has been created yet, we must initialize one
            initiate_optimizer_flag = True
        else:
            # Assume no need to init; will override if other conditions apply
            initiate_optimizer_flag = False

        # Update hyperparameters if provided, and re-init optimizer accordingly
        if hyperparam is not None:
            self.hyperparam.update(hyperparam)
            initiate_optimizer_flag = True   # new settings require re-init

        # Override optimizer class if explicitly provided, and re-init
        if optimizer_class is not None:
            self.optimizer_class = optimizer_class
            initiate_optimizer_flag = True

        # (Re)initialize the optimizer if flagged
        if initiate_optimizer_flag:
            # For initiating optimizer, get model parameters (grouped or flat)
            net_ = self._model.net_
            if '_groups' in net_.__dict__.keys():
                params = net_.grouped_parameters()
            else:
                params = net_.parameters()

            # Create the optimizer using the selected optimizer class
            self.optimizer = self.optimizer_class(params, **self.hyperparam)

        # Setup scheduler if provided
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)

        # Setup alpha scheduler if alpha_tmax is provided
        if alpha_tmax is not None:
            self.alpha_scheduler = AlphaScheduler(alpha_tmax)

        # Update checkpoint configuration if provided
        if checkpoint_dict is not None:
            self.checkpoint_dict.update(checkpoint_dict)

        # Update the loss function if provided
        if loss_func is not None:
            self.loss_func = loss_func

    def execute(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        **setup_kwargs
    ):
        """
        Executes the training loop with the specified configuration.

        Parameters
        ----------
        n_epochs : int, optional, default=100
            Number of training epochs.

        batch_size : int, optional, default=64
            Size of training batches.

        **setup_kwargs : dict
            Additional keyword arguments passed directly to
            `setup_optimizer_and_components`. These can include:

            - optimizer_class
            - scheduler
            - loss_func
            - alpha_tmax
            - hyperparam
            - checkpoint_dict

        Notes
        -----
        Delegates configuration and setup of training components to
        `setup_optimizer_and_components()`.
        """
        self.setup_optimizer_and_components(**setup_kwargs)

        # Begin training if n_epochs > 0
        if n_epochs > 0:
            self._train(n_epochs, batch_size)

    def execute_ddp_training(self, seeds_list=None, **train_kwargs):
        """
        Execute distributed training using Distributed Data Parallel (DDP).

        Here are the steps:
        1. Initialize the process group for distributed communication.
        2. Wrap the model with DDP for multi-GPU training.
        3. Set random seeds for reproducibility.
        4. Execute the training routine.
        5. Synchronize all processes.
        6. Destroy the process group to free resources.
        """
        # Initialize distributed backend
        self._model.device_handler.init_process_group(backend="nccl")
        self._model.device_handler.ddp_wrapper()
        self._model.device_handler.set_seed(seeds_list)

        # Log initialization
        logging.info("Process group initialized & model wrapped with DDP.")

        # Execute training
        self.execute(**train_kwargs)

        # Synchronize processes after training
        torch.distributed.barrier()

        # Ensure cleanup of the process group
        self._model.device_handler.destroy_process_group()
        logging.info("Process group destroyed.")

    def _train(self, n_epochs, batch_size):
        """Train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training

        batch_size : int
            Size of samples used at each epoch
        """

        self.train_history['ess'].extend([None] * n_epochs)
        self.train_history['loss'].extend([None] * n_epochs)
        self.train_history['logp'].extend([None] * n_epochs)
        self.train_history['logqp'].extend([None] * n_epochs)

        rank = self._model.device_handler.rank

        last_epoch = self.train_history['epoch']
        report_progress = self.checkpoint_dict['print_every'] is not None

        if last_epoch == 0:
            self._checkpoint(last_epoch, None, None)

        if rank == 0 and report_progress:
            print(f">>> Training started for {n_epochs} epochs <<<")

        t_1 = time.time()

        for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):

            loss, logq, logp = self.step(batch_size)
            self._checkpoint(epoch, logq, logp)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()

        t_2 = time.time()

        if rank == 0 and report_progress:
            print(f">>> Training finished ({loss.device});", end='')
            print(f" TIME = {t_2 - t_1:.3g} sec <<<")

    def step(self, batch_size, debug=False):
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
        model = self._model
        prior = model.prior

        # Sample inputs from the prior
        x, logr = prior.sample_(batch_size)

        # Forward pass through the neural network
        y, logj = model.net_.forward(x)

        # Compute the log-probability of the transformed data `y`
        logq = logr - logj

        if self.path_gradient_autodiff:
            logq += adjustment_for_path_gradient_autodiff(y, model.net_, prior)

        # Compute target log-probability
        if self.alpha_scheduler is None:
            logp = - model.action(y)
        else:
            alpha = self.alpha_scheduler.alpha
            logp = - alpha * model.action(y) + (1 - alpha) * prior.log_prob(y)

        # Compute Loss
        loss = self.loss_func(logq, logp)

        # Backpropagation and optimization
        self.optimizer.zero_grad()  # clears old gradients from last steps
        loss.backward()
        self.optimizer.step()

        if debug:
            param = list(model.net_.parameters())[-1]
            print((
                f"rank = {self._model.device_handler.rank} | "
                f"loss = {loss.item():.4f} | "
                f"param = {param.ravel()[0].item():.14e} | "
                f"param.grad = {param.grad.ravel()[0].item():.14e}\n"
            ))

        return loss, logq, logp

    @torch.no_grad()
    def _checkpoint(self, epoch, logq, logp):
        """
        Computes training metrics and updates the training history during model
        training. This method logs metrics every `print_every` epochs. Only
        the process with rank 0 logs and stores the metrics.

        Parameters:
        ----------
        epoch : int
            The current epoch in the training loop.

        logq : torch.Tensor
            Log-probabilities under the model.

        logp : torch.Tensor
            Log-probabilities under the target distribution.
        """

        every = self.checkpoint_dict['print_every']
        bsize = self.checkpoint_dict['print_bsize']

        if every is not None and epoch % every == 0:
            # Compute metrics and log at the specified print stride
            out = self.compute_metrics(
                logq=logq,
                logp=logp,
                epoch=epoch,  # the metrics will be printed for integer epoch
                batch_size=bsize  # generates new samples if bsize is not None
            )
        else:
            # Otherwise, only compute metrics
            out = self.compute_metrics(logq=logq, logp=logp)

        if epoch == 0:
            return

        # Update the training history if on rank 0
        if self._model.device_handler.rank == 0:
            loss, ess, logqp, logp = out
            self.train_history['epoch'] = epoch
            self.train_history['ess'][epoch - 1] = ess
            self.train_history['loss'][epoch - 1] = loss
            self.train_history['logp'][epoch - 1] = logp
            self.train_history['logqp'][epoch - 1] = logqp
        else:
            # out is None for non-zero ranks, and we do not update history.
            pass

    @torch.no_grad()
    def compute_metrics(
        self, batch_size=None, logq=None, logp=None, epoch=None
    ):
        """
        Computes training metrics such as loss, effective sample size (ESS),
        and log-probabilities. Optionally logs the metrics if `epoch` is
        provided as an iteger.

        This includes:
            - Loss: Computed using the model's loss function.
            - ESS (Effective Sample Size): Measures sample quality.
            - log(q/p): Mean and standard deviation of the log-ratio of model
              and target densities.
            - log(p): Mean and standard deviation of log-probabilities under
              the target distribution.

        Parameters:
        -----------
        batch_size : int | None, optional
            If `batch_size` is provided, both `logq` and `logp` are ignored,
            and the method will sample `logq` and `logp` using the model.

        logq : torch.Tensor | None, optional
            Log-probabilities under the model. Ignored if `batch_size` is
            provided.

        logp : torch.Tensor | None, optional
            Log-probabilities under the target distribution. Ignored if
            `batch_size` is provided.

        epoch : int | None, optional
            If an integer, logs the metrics to the console for this epoch.
            If `None`, no logging is performed.

        Returns:
        --------
        tuple:
            - loss (float): The training loss.
            - ess (float): Effective sample size.
            - logqp (tuple): Mean and standard deviation of log(q/p).
            - logp (tuple): Mean and standard deviation of log(p).
        """
        # Sample logq and logp if `batch_size` is provided
        if batch_size is not None:
            _, logq, logp = self._model.posterior.sample__(batch_size)

        elif (logq is None or logp is None):
            return None

        # Gather logq and logp across devices for distributed setups
        logq = self._model.device_handler.all_gather_into_tensor(logq)
        logp = self._model.device_handler.all_gather_into_tensor(logp)

        # Terminate if not on zero rank
        if self._model.device_handler.rank > 0:
            return None

        # Compute metrics
        loss = self.loss_func(logq, logp).item()  # Compute loss
        ess = self.calc_ess(logq, logp).item()  # Compute effective sample size

        # Compute the mean & std of log(q/p), which is called logqp, and log(p)
        logqp = ((logq - logp).mean().item(), (logq - logp).std().item())
        logp = (logp.mean().item(), logp.std().item())

        # Log metrics if epoch is an integer
        if isinstance(epoch, int):
            log_message = (
                f"Epoch: {epoch} | loss: {loss:.4f} | ess: {ess:.4f} | "
                f"log(q/p): {fmt_val_err(*logqp, err_digits=2)} | "
                f"log(p): {fmt_val_err(*logp, err_digits=2)}"
            )
            print(log_message)

        # Return computed metrics
        return loss, ess, logqp, logp

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

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the state of the training process.
        """
        checkpoint = {
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            # 'alpha_state_dict': self.alpha_scheduler.state_dict(),
            'epoch': self.train_history['epoch'],
            'loss': self.train_history['loss'],
            'ess': self.train_history['ess'],
            'logp': self.train_history['logp']
        }

        return checkpoint

    def load_state_dict(self, checkpoint: dict):
        """Loads the training state from a checkpoint dictionary.

        AssertionError:
            Always raised, as the method is not yet implemented.
        """

        # model should be loaded to cpu and then moved to GPUs if needed

        assert False, "OOPS: this is not implemeted yet! Try Later!"

        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.alpha_scheduler.load_state_dict(checkpoint['alpha_state_dict'])
        self.train_history['epoch'] = checkpoint['epoch']
        self.train_history['loss'] = checkpoint['loss']
        self.train_history['ess'] = checkpoint['ess']
        self.train_history['logp'] = checkpoint['logp']


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
    d_logj = (minus_logj - minus_logj.detach())
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
@torch.no_grad()
def reverse_flow_sanitychecker(model, n_samples=4, net_=None):
    """Performs a sanity check on the reverse method of modules."""

    if net_ is None:
        net_ = model.net_

    x = model.prior.sample(n_samples)
    y, logj = net_(x)
    x_hat, minus_logj = net_.reverse(y)

    def mean(z):
        return z.abs().mean().item()

    print("reverse mode is OK if following values vanish (up to round off):")
    print(f"{mean(x - x_hat):g} & {mean(torch.exp(logj + minus_logj) - 1):g}")
