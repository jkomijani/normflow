# Copyright (c) 2021-2024 Javad Komijani

"""
This module contains high-level classes for normalizing flow techniques,
with the central `Model` class integrating essential components such as priors,
networks, and actions. It provides utilities for training and sampling,
along with support for MCMC sampling and device management.
"""

import torch
import time

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
    fit : Fitter
        An instance of the Fitter class, responsible for training the model.
        `fit` is aliased to `train` for flexibility in usage.

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

        self.net_ = net_
        self.prior = prior
        self.action = action

        # Components for training, sampling, and device handling
        self.fit = Fitter(self)
        self.train = self.fit  # Alias for `fit`

        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)
        self.device_handler = ModelDeviceHandler(self)

        if load_checkpoint_path is not None:
            self.load_checkpoint(load_checkpoint_path)

    def state_dict(self):

        return {'net_state_dict': self.net_.state_dict(),
                'prior_state_dict': {},  # self.prior.state_dict(),
                'action_state_dict': {},  # self.action.state_dict(),
                'train_state_dict': self.train.state_dict()
                }

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(
            self,
            path: str,
            parameters_only: bool = True,
            map_location = torch.device('cpu')
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

        if parameters_only == False:
            self.prior.load_state_dict(state['prior_state_dict'])
            self.action.load_state_dict(state['action_state_dict'])
            self.train.load_state_dict(state['train_state_dict'])


# =============================================================================
class Posterior:
    """
    Creates samples directly from a trained probabilistic model.

    The `Posterior` class generates samples from a specified model without
    using an accept-reject step, making it suitable for tasks that require
    quick, direct sampling. All methods in this class use `torch.no_grad()`
    to prevent gradient computation.

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

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
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
class Fitter:
    """A class for training a given model."""

    path_gradient_autodiff = True

    def __init__(self, model: Model):
        self._model = model

        self.train_history = dict(
                epoch=0, loss=[None], ess=[None], logp=[None]
                )

        self.hyperparam = dict(lr=0.001, weight_decay=0.01)

        self.checkpoint_dict = dict(
            print_stride=10,
            print_batch_size=None
            )

    def __call__(self,
            n_epochs=1000,
            batch_size=64,
            optimizer_class=torch.optim.AdamW,
            scheduler=None,
            loss_fn=None,
            alpha_tmax=None,
            hyperparam={},
            checkpoint_dict={}
            ):

        """Fit the model; i.e. train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training.

        batch_size : int
            Size of samples used at each epoch.

        optimizer_class : optimization class, optional
            By default is set to torch.optim.AdamW, but can be changed.

        scheduler : scheduler class, optional
            By default no scheduler is used.

        loss_fn : None or function, optional
            The default value is None, which translates to using KL divergence.

        alpha_tmax : int or None, optional
            If a positive integer, a scheduler would be setup for interpolating
            between prior and target distributions. Default is None.
            (See `AlphaScheduler`.)

        hyperparam : dict, optional
            Can be used to set hyperparameters like the learning rate and decay
            weights.

        checkpoint_dict : dict, optional
            Can be set to control printing the status of the training.
        """
        self.hyperparam.update(hyperparam)
        self.checkpoint_dict.update(checkpoint_dict)

        self.loss_fn = Fitter.calc_kl_mean if loss_fn is None else loss_fn

        net_ = self._model.net_
        if '_groups' in net_.__dict__.keys():
            parameters = net_.grouped_parameters()
        else:
            parameters = net_.parameters()
        self.optimizer = optimizer_class(parameters, **self.hyperparam)

        if scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer)

        self.alpha_scheduler = AlphaScheduler(alpha_tmax)

        if n_epochs > 0:
            self._train(n_epochs, batch_size)

    def _train(self, n_epochs, batch_size):
        """Train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training

        batch_size : int
            Size of samples used at each epoch
        """

        last_epoch = self.train_history['epoch']
        if last_epoch == 0:
            if self._model.device_handler.rank == 0:
                print(f">>> Checking the current status of the model <<<")
            self._checkpoint(last_epoch, None, None, batch_size)

        self.train_history['ess'].extend([None] * n_epochs)
        self.train_history['loss'].extend([None] * n_epochs)
        self.train_history['logp'].extend([None] * n_epochs)

        rank = self._model.device_handler.rank

        if rank == 0:
            print(f">>> Training started for {n_epochs} epochs <<<")

        t_1 = time.time()
        for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):
            loss, logq, logp = self.step(batch_size)
            self._checkpoint(epoch, logq, logp)
            if self.scheduler is not None:
                self.scheduler.step()
            self.alpha_scheduler.step()
        t_2 = time.time()

        if rank == 0:
            print(f">>> Training finished ({loss.device});", end='')
            print(f" TIME = {t_2 - t_1:.3g} sec <<<")

    def step(self, batch_size):
        """Perform a train step with a batch of inputs of size `batch_size`."""
        model = self._model
        alpha = self.alpha_scheduler.alpha

        x, logr = model.prior.sample_(batch_size)
        y, logj = model.net_(x)
        logp = - alpha * model.action(y)
        logq = alpha * logr - logj

        if self.path_gradient_autodiff:
            x, minus_logj = model.net_.reverse(y.detach())
            logq_ydetached = minus_logj + alpha * model.prior.log_prob(x)
            logq = logq - (logq_ydetached - logq_ydetached.detach())

        loss = self.loss_fn(logq, logp)

        self.optimizer.zero_grad()  # clears old gradients from last steps

        loss.backward()

        self.optimizer.step()

        return loss, logq, logp

    def print_trian_history(self, keys=['loss', 'ess', 'logp']):
        for key, value in self.train_history.items():
            print(key)
            print(value)

    @torch.no_grad()
    def _checkpoint(self, epoch, logq, logp, batch_size=None):

        rank = self._model.device_handler.rank

        stride = self.checkpoint_dict['print_stride']

        # Generate new samples if demanded!
        if epoch % stride == 0:

            bsize = self.checkpoint_dict['print_batch_size']

            if (bsize is None) and (logq is not None):
                pass  # use the input logq & logp
            else:
                # draw samples to calculate logq & logp
                bsize_ = batch_size if (bsize is None) else bsize
                _, logq, logp = self._model.posterior.sample__(bsize_)

        logq = self._model.device_handler.all_gather_into_tensor(logq)
        logp = self._model.device_handler.all_gather_into_tensor(logp)

        if rank == 0:
            ess = self.calc_ess(logq, logp).item()
            loss = self.loss_fn(logq, logp).item()
            logp = (logp.mean().item(), logp.std().item())

            if epoch % stride == 0:
                str1 = f"Epoch: {epoch} | loss: {loss:.4f} | ess: {ess:.4f} | "
                str2 = "log(p): {0}".format(fmt_val_err(*logp, err_digits=2))
                print(str1 + str2)

            self.train_history['epoch'] = epoch
            self.train_history['ess'][epoch] = ess
            self.train_history['loss'][epoch] = loss
            self.train_history['logp'][epoch] = logp

        return

    @staticmethod
    def calc_kl_mean(logq, logp):
        """Return Kullback-Leibler divergence estimated from logq and logp."""
        return (logq - logp).mean()  # KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq, logp):
        return (logq - logp).var()

    @staticmethod
    def calc_corrcoef(logq, logp):
        return torch.corrcoef(torch.stack([logq, logp]))[0, 1]

    @staticmethod
    def calc_direct_kl_mean(logq, logp):
        logpq = logp - logq
        logz = torch.logsumexp(logpq, dim=0) - np.log(logp.shape[0])
        logpq = logpq - logz  # p is now normalized
        p_by_q = torch.exp(logpq)
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_minus_logz(logq, logp):
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

    def state_dict(self):

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

    def load_state_dict(self, checkpoint):

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
class AlphaScheduler:
    """Introduces a parameter and a scheduler to change it."""

    def __init__(self, t_max=None):

        self.alpha = 1 if t_max is None else 0
        self.t_max = t_max

    def step(self):
        if not (self.t_max is None):
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

    mean = lambda z: z.abs().mean().item()

    print("reverse mode is OK if following values vanish (up to round off):")
    print(f"{mean(x - x_hat):g} & {mean(torch.exp(logj + minus_logj) - 1):g}")
