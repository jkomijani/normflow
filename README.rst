normflow
--------
This package contains utilities for the implementation of the method of
normalizing flows as a generative model for lattice field theory.

The method of normalizing flows is a powerful approach in generative modeling
that aims to learn complex probability distributions by transforming samples
from a simple distribution through a series of invertible transformations.
It has found applications in various domains, including generative image
modeling.

The package currently supports scalar theories in any dimension, and we are
actively extending the package to accommodate gauge theories, broadening its
applicability.

In a nutshell, for the method of normalizing flows, one should provide three
essential components:

*   a prior distribution to draw initial samples,
*   a neural network to perform a series of invertible transformations on the
    samples,
*   an action that specifies the target distribution, defining the goal of the
    generative model.

The central high-level class of the package is called ``Model``, which can be
instantiated by providing instances of the three objects mentioned earlier for
the prior, the neural network, and the action.

Following the terminology used by scikit-learn, every instance of ``Model`` is
equipped with a method called ``fit`` that facilitates the training of the model.
Training involves optimizing the parameters of the neural network to achieve a
transformation that effectively maps the prior distribution to the target
distribution.

Below is a simple example of a scalar theory in zero dimension:

.. code-block::

    from normflow import Model
    from normflow.action import ScalarPhi4Action
    from normflow.prior import NormalPrior
    from normflow.nn import DistConvertor_

    def make_model():
        prior = NormalPrior(shape=(1,))
        action = ScalarPhi4Action(kappa=0, m_sq=-2.0, lambd=0.2)
        net_ = DistConvertor_(knots_len=10, symmetric=True)
        model = Model(net_=net_, prior=prior, action=action)
        return model

    model = make_model()
    model.fit(n_epochs=1000, batch_size=1024, checkpoint_dict=dict(print_stride=100))

The above code block results in an output similar to::

    >>> Checking the current status of the model <<<
    Epoch: 0 | loss: -1.8096 | ess: 0.4552 | log(p): 0.4(11)
    >>> Training started for 1000 epochs <<<
    Epoch: 100 | loss: -2.0154 | ess: 0.6008 | log(p): 0.52(57)
    Epoch: 200 | loss: -2.1092 | ess: 0.7381 | log(p): 0.60(56)
    Epoch: 300 | loss: -2.1612 | ess: 0.8195 | log(p): 0.57(87)
    Epoch: 400 | loss: -2.2091 | ess: 0.8783 | log(p): 0.63(83)
    Epoch: 500 | loss: -2.2459 | ess: 0.9262 | log(p): 0.71(58)
    Epoch: 600 | loss: -2.2670 | ess: 0.9459 | log(p): 0.73(56)
    Epoch: 700 | loss: -2.2684 | ess: 0.9585 | log(p): 0.74(53)
    Epoch: 800 | loss: -2.2667 | ess: 0.9684 | log(p): 0.74(51)
    Epoch: 900 | loss: -2.2724 | ess: 0.9789 | log(p): 0.76(54)
    Epoch: 1000 | loss: -2.2673 | ess: 0.9791 | log(p): 0.75(62)
    >>> Training finished (cpu); TIME = 4.36 sec <<<


After training the model, one can draw samples using an attribute called
``posterior``; to draw ``n`` samples from the trained distribution, use:

.. code-block::

    x = model.posterior.sample(n)

Note that the train distribution is almost never identical to the target
distribution, which is specified by the action.
To generate samples that are correctly drawn from the target distribution,
similar to Markov Chain Monte Carlo (MCMC) simulations,
one can employ a Metropolis accept/reject step and discard some of the first
samples; to this end, one can use:

.. code-block::

    x = model.mcmc.sample(n)

which draws ``n`` samples from the trained distribution and applies a
Metropolis accept/reject step to ensure that the samples are correctly drawn.

Moreover, the model has an attribute called ``device_handler``, which can be
used to specify the number of GPUs used for training (the default value is one
if any GPU is available).
To this end, one can use:

.. code-block::

    def fit_func(model):
        model.fit(n_epochs=500, batch_size=128)

    model.device_handler.spawnprocesses(fit_func, nranks)

where ``nranks`` specifies the number of GPUs.


| Created by Javad Komijani on 2021
| Copyright (C) 2021-24, Javad Komijani
