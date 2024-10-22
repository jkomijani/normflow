# Copyright (c) 2021-2024 Javad Komijani

"""This is a module containing the core components for normalizing flow.

The central high-level class is called Model, which takes instances of
other classes as input (`prior`, `net_`, and `action`) and provides untilities
to perform training and drawing samples.

Every instance of the central high-level class Model alreay has an instance of
Fitter, which can be used for training.

For drawing samples, one can use `posterior`, which does not perform any
Metropolis accept/reject on the samples, or one can use `mcmc` if Metropolis
accept/reject needed.

Other central classes in this module are `Module_`, and `ModuleList_`
that allow us to define neural networks; these two classes are imported and
used by other modules of this package.
"""


import torch
import time

import numpy as np

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import fmt_val_err
from .device import ModelDeviceHandler


# =============================================================================
class Model:
    """The central high-level class of the package, which
    takes instances of other classes as input (prior, net_, and action)
    and provides untilities to perform training and drawing samples.

    Parameters
    ----------
    prior : An instance of a Prior class (e.g NormalPrior).

    net_ : An instance of ModuleList_ or similar classes. The trailing
        underscore implies that the associate forward method handles
        the Jacobian of the transformation.

    action : An instance of a class that describes the action.

    name : str, option
        A string to label the model
    """

    def __init__(self, *, prior, net_, action, name=None):
        self.name = name
        self.net_ = net_
        self.prior = prior
        self.action = action

        self.fit = Fitter(self)
        self.train = self.fit  # an alias for fit

        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)
        self.device_handler = ModelDeviceHandler(self)

    def transform(self, x):
        return self.net_(x)[0]


class Posterior:
    """A class for drawing samples from given model. Note that the samples
    are drawn directly from the model without performing any accept/reject
    filtering.

    Parameters
    ----------
    model : An instance of Model
    """

    def __init__(self, model):
        self._model = model

    @torch.no_grad()
    def sample(self, batch_size=1, **kwargs):
        return self.sample_(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(self, batch_size=1, preprocess_func=None):
        """
        Return `batch_size` samples along with `log(q)` and `log(p)`.

        Parameters
        ----------
        batch_size: int
            The size of the samples

        preprocess_func: None or a function
            Introduced to preprocess the prior sample if needed
        """
        x, logr = self._model.prior.sample_(batch_size)
        if preprocess_func is not None:
            x, logr = preprocess_func(x, logr)
        y, logj = self._model.net_(x)
        logq = logr - logj
        return y, logq

    @torch.no_grad()
    def sample__(self, batch_size=1, **kwargs):
        y, logq = self.sample_(batch_size=batch_size, **kwargs)
        logp = -self._model.action(y)  # logp is log(p * z)
        return y, logq, logp

    # @torch.no_grad()
    # The `no_grad` is removed for use with `path_gradient_autodiff`
    def log_prob(self, y):
        """Returns log probability of the samples."""
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

        self.train_history = dict(loss=[None], ess=[None], logp=[None])

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

        initial_epoch = len(self.train_history['loss'])
        if initial_epoch == 1:
            if self._model.device_handler.rank == 0:
                print(f">>> Checking the current status of the model <<<")
            self._checkpoint(0, None, None, batch_size)

        self.train_history['ess'].extend([None] * n_epochs)
        self.train_history['loss'].extend([None] * n_epochs)
        self.train_history['logp'].extend([None] * n_epochs)

        rank = self._model.device_handler.rank

        if rank == 0:
            print(f">>> Training started for {n_epochs} epochs <<<")

        t_1 = time.time()
        for epoch in range(initial_epoch, initial_epoch + n_epochs):
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

    print("reverse method is OK if following values vanish (up to round off):")
    print(f"{mean(x - x_hat):g} & {mean(1 + minus_logj / logj):g}")
