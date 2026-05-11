"""Import all modules of `nn` subpackage"""

# Core modules:
from ._core import (
    Module_,
    ModuleList_,
    MultiChannelModule_,
    MultiOutChannelModule_,
    InvisibilityMaskWrapperModule_
)

from .unet_ import *

from .scalar.modules import *

from .scalar.modules_ import *
from .scalar.couplings_ import *

from .scalar.time_embedding import *
from .scalar.rqs_modules_ import *

# Special transformations:
from .scalar.auto_regressive_ import *

from .scalar.fftflow_ import *
from .scalar.meanfield_ import *
from .scalar.psd_ import *

from .scalar.planar_ import MultiPlanarFlow_

from .matrix.matrix_module_ import MatrixModule_
from .matrix.stapled_matrix_module_ import StapledMatrixModule_

# SU(N) gauge modules:
from .gauge.planar_gauge_module_ import PlanarGaugeModule_
from .gauge.planar_gauge_module_ import PlanarGaugeModuleList_

from .gauge.gauge_module_ import *

from .gauge.gauge_param_couplings_ import (
    Pade11Coupling_,
    Pade22Coupling_,
    SU3RQSplineCoupling_,
    SU2RQSplineCoupling_,
    U1RQSplineCoupling_,
    SUnParamAffineCoupling_
)

from .gauge.gauge_param_dual_couplings_ import (
    Pade11DualCoupling_,
    Pade22DualCoupling_,
    SU2RQSplineDualCoupling_,
    SU3RQSplineDualCoupling_
)

from .gauge.trivializing_map_ import WilsonTrivMap_
from .gauge.unitary_flow_ import ModalMatrixSteppedCommutatorFlow_
from .gauge.modal_commutator_odeflow_ import ModalCommutatorFlow_
