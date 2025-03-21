# Base modules:
from ._core import Module_, ModuleList_
from ._core import MultiChannelModule_, MultiOutChannelModule_
from ._core import InvisibilityMaskWrapperModule_

from .scalar.modules import AvgNeighborPool
from .scalar.modules import ConvBlock
from .scalar.modules import DenseBlock
from .scalar.modules import Affine
from .scalar.modules import Pade32


# Subclasses of Module_
from .scalar.modules_ import DistConvertor_, Identity_, Clone_
from .scalar.modules_ import UnityDistConvertor_, PhaseDistConvertor_
from .scalar.modules_ import Affine_
from .scalar.modules_ import Pade11_, Pade22_, Pade32_, Pade32a_

# Coupling layers:
from .scalar.couplings_ import ShiftCoupling_, AffineCoupling_
from .scalar.couplings_ import RQSplineCoupling_, MultiRQSplineCoupling_


# Special transformations:

from .scalar.planar_ import MultiPlanarFlow_

from .scalar.auto_regressive_ import FiboARModule_
from .scalar.auto_regressive_ import VectorARModule_

from .scalar.fftflow_ import FFTNet_
from .scalar.meanfield_ import MeanFieldNet_
from .scalar.psd_ import PSDBlock_

from .matrix.matrix_module_ import MatrixModule_
from .matrix.stapled_matrix_module_ import StapledMatrixModule_

from .gauge.planar_gauge_module_ import PlanarGaugeModule_
from .gauge.planar_gauge_module_ import PlanarGaugeModuleList_

from .gauge.gauge_module_ import GaugeModule_
from .gauge.gauge_module_ import GaugeModuleList_
from .gauge.gauge_module_ import PolyakovGaugeModule_

from .gauge.gauge_param_couplings_ import Pade11Coupling_
from .gauge.gauge_param_couplings_ import Pade22Coupling_
from .gauge.gauge_param_couplings_ import SU3RQSplineCoupling_
from .gauge.gauge_param_couplings_ import SU2RQSplineCoupling_
from .gauge.gauge_param_couplings_ import U1RQSplineCoupling_
from .gauge.gauge_param_couplings_ import SUnParamAffineCoupling_

from .gauge.gauge_param_dual_couplings_ import Pade11DualCoupling_
from .gauge.gauge_param_dual_couplings_ import Pade22DualCoupling_
from .gauge.gauge_param_dual_couplings_ import SU2RQSplineDualCoupling_
from .gauge.gauge_param_dual_couplings_ import SU3RQSplineDualCoupling_


try:
    from .gauge.trivializing_map_ import WilsonTrivMap_
    from .gauge.unitary_flow_ import ModalMatrixFlow_
except:
    pass
