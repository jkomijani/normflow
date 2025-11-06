# Copyright (c) 2024-2025 Javad Komijani


from lattice_ml.integrate import AdjLieODEFlow_
from lattice_ml.integrate import AdjLieModule


class WilsonTrivMap_(AdjLieODEFlow_):
    """
    Trivializing map for Wilson gauge action based on Luscher's construction
    (arXiv:0907.5491).

    Args:
        wilson_action: The Wilson gauge action from `normflow.action`.
        t_span (int, int): Integration interval in flow time (default: (0, 1)).
        step_size (float): Integration step size. Defaults to `0.1 / beta`.
        order (str): Must be 'LO'. Higher orders are not yet implemented.
        **solver_kwargs: Additional keyword arguments passed to the ODE solver.
    """

    def __init__(
        self,
        wilson_action,
        t_span=(0, 1),
        step_size=None,
        order='LO',
        **solver_kwargs
    ):
        assert order == 'LO', "NLO is not implemented yet"

        func = WilsonFlowDynamics(wilson_action)

        if step_size is None:
            # Empirical default: 0.1 works well for beta = 1
            step_size = 0.1 / wilson_action.beta

        solver_kwargs['step_size'] = step_size
        super().__init__(func, t_span=t_span, **solver_kwargs)


class WilsonFlowDynamics(AdjLieModule):
    """
    Lie-algebra-valued dynamics for the Wilson flow at LO in the trivializing
    map construction.

    Implements the dynamics (right hand side) of following ODE:
        dA_t/dt = F(A_t) / (4 * C_F)
    where F(A_t) is the algebra force from the Wilson action and
    C_F is the quadratic Casimir in the Fandamental representation.
    """
    # - Luscher uses Tr(T^a T^b) = -1/2 δ^ab, giving
    #      C_F = (N_c^2 - 1) / (2 N_c).
    # - In `action.algebra_force` this code uses Tr(T^a T^b) = -δ^ab;
    #   therefore, our Casimir is
    #      C_F = (N_c^2 - 1) / N_c.
    # - The flow equation is unchanged in form but uses the correct C_F value.

    def __init__(self, wilson_action):
        super().__init__()
        self.wilson_action = wilson_action
        n_c = wilson_action.n_c
        # Casimir constant for our normalization used in action.algebra_force
        self.c_f = (n_c ** 2 - 1) / n_c

    def algebra_dynamics(self, t, x):
        """
        Compute the algebra-valued time derivative at flow time t.

        Args:
            t (float): Flow time.
            x (torch.Tensor): Gauge-field configuration.

        Returns
            torch.Tensor: The Lie-algebra-valued dynamics.
        """
        force = self.wilson_action.algebra_force(x)
        return force / (4 * self.c_f)

    def calc_logj_rate(self, t, x):
        """
        Compute the rate of change of the log-Jacobian determinant.

        Args:
            t (float): Flow time.
            var (array-like): Gauge-field configuration.

        Returns:
            float: Jacobian rate: :math:`d/dt \log(det J_t)`.
        """
        # At leading order, the Jacobian rate equals the action value
        # (see Eq. (3.9) in arXiv:0907.5491).
        return self.wilson_action(x)
