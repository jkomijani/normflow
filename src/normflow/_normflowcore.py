# Copyright (c) 2021-2026 Javad Komijani

"""
This module contains high-level classes for normalizing flow techniques,
with the central `Model` class integrating essential components such as priors,
networks, and actions. It provides utilities for training and sampling,
along with support for MCMC sampling and device management.
"""

import torch

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import fmt_val_err
from ._trainer import Trainer


__all__ = ["Model", "reverse_flow_sanitychecker"]


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
    """

    def __init__(self, *, prior, net_, action):

        # Main components of the model
        self.net_ = net_
        self.prior = prior
        self.action = action

        # Components for training
        self.trainer = Trainer(self)
        self.train = self.trainer.run_training  # alias
        self.fit = self.trainer.run_training  # another alias

        # Components for sampling
        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)

    @torch.no_grad()
    def compute_metrics(self, batch_size: int, epoch: int | None = None):
        """
        Computes effective sample size (ESS) and log-probabilities.
        Optionally logs the metrics if `epoch` is provided as an iteger.

        Returns:
        --------
        tuple:
            - ess (float): Effective sample size.
            - logqp (tuple): Mean and standard deviation of log(q/p).
            - logq (tuple): Mean and standard deviation of log(p).
            - logp (tuple): Mean and standard deviation of log(p).
        """
        _, logq, logp = self.posterior.sample__(batch_size)

        # Compute effective sample size
        ess = calc_ess(logq, logp).item()

        # Compute the mean & std of log-probabilites
        logqp = ((logq - logp).mean().item(), (logq - logp).std().item())
        logq = (logq.mean().item(), logq.std().item())
        logp = (logp.mean().item(), logp.std().item())

        # Log metrics if epoch is an integer
        if isinstance(epoch, int):
            log_message = (
                f"Epoch: {epoch} | ess: {ess:.4f} | "
                f"log(q/p): {fmt_val_err(*logqp, err_digits=2)} | "
                f"log(q): {fmt_val_err(*logq, err_digits=2)} | "
                f"log(p): {fmt_val_err(*logp, err_digits=2)}"
            )
            print(log_message)

        # Return computed metrics
        return ess, logqp, logq, logp


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
def calc_ess(logq, logp):
    """Rerturn effective sample size (ESS)."""
    logpq = logp - logq
    log_ess = 2*torch.logsumexp(logpq, dim=0) - torch.logsumexp(2*logpq, dim=0)
    ess = torch.exp(log_ess) / len(logpq)  # normalized
    return ess


@torch.no_grad()
def reverse_flow_sanitychecker(model, n_samples=4, net_=None):
    """Performs a sanity check on the reverse method of modules."""

    if net_ is None:
        net_ = model.net_

    x = model.prior.sample(n_samples)
    y, logj = net_(x)
    x_hat, minus_logj = net_.reverse(y)

    jac_product = torch.exp(logj + minus_logj).cpu().numpy()
    diff = (x - x_hat).abs().reshape(n_samples, -1).sum(dim=1).cpu().numpy()
    norm = x.abs().reshape(n_samples, -1).sum(dim=1).cpu().numpy()

    print(f"The test is performed on {n_samples} samples:")
    print("|x - f⁻¹f(x)| / |x| =", diff / norm)
    print("log ({J_f}⁻¹)       =", -logj.cpu().numpy())
    print("log (J_{f⁻¹})       =", minus_logj.cpu().numpy())
    print("J_f * J_{f⁻¹}   = 1 +", jac_product - 1)
