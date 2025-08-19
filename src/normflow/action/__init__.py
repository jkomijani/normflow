# List of supported actions

from .gauge_action import WilsonGaugeAction, U1WilsonGaugeAction
GaugeAction = WilsonGaugeAction  # alias for legacy, will be removed later
U1GaugeAction = U1WilsonGaugeAction  # alias for lagacy, will be removed later

# from .schwinger_action import SchwingerAction

from .ginibre_gauge_action import GinibreGaugeAction

from .matrix_action import MatrixAction

from .scalar_action import ScalarPhi4Action
