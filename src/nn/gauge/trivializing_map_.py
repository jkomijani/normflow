# Copyright (c) 2024 Javad Komijani


from torch_solve_ext.integrate import AdjLieODEflow_
from torch_solve_ext.integrate import AdjLieModule


class WilsonTrivMap_(AdjLieODEflow_):

    def __init__(
            self, wilson_action, t_span=[0, 1], step_size=None, order='LO'
            ):

        assert order == 'LO', "NLO is not implemented yet"

        func = WilsonFlowDynamics(wilson_action)

        if step_size is None:
            step_size = 0.1 / wilson_action.beta  # 0.1 is good for beta = 1

        super().__init__(func, t_span=t_span, step_size=step_size)


class WilsonFlowDynamics(AdjLieModule):

    def __init__(self, wilson_action):
        super().__init__()
        self.wilson_action = wilson_action
        n_c = wilson_action.n_c
        self.c_f = (n_c ** 2 - 1) / n_c  # (c_f = 8 / 3) for SU(3)

    def algebra_dynamics(self, t, var):
        force = self.wilson_action.algebra_force(var)
        return force / (2 * self.c_f)

    def calc_logj_rate(self, t, var):
        # see e.g. (3.9) in [arXiv:0907.5491] or use `super().calc_logj_rate`.
        return self.wilson_action(var)
