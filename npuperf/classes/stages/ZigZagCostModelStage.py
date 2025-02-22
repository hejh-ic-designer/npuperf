from typing import Generator, Callable, List, Tuple, Any
from npuperf.classes.stages.Stage import Stage
import logging
from npuperf.classes.cost_model.cost_model import CostModelEvaluation
from npuperf.classes.cost_model.mem_engine import MemoryOperatorEvaluation
from npuperf.classes.workload.layer_node import LayerNode
from npuperf.classes.workload.mem_node import MemNode
logger = logging.getLogger(__name__)


class ZigZagCostModelStage(Stage):
    """
    Pipeline stage that calls a cost model to evaluate a mapping on a HW config.
    """
    def __init__(self, list_of_callables:List[Callable], *, accelerator, layer, spatial_mapping=None, temporal_mapping=None, **kwargs):
        """
        Initializes the cost model stage given main inputs
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping, self.temporal_mapping =\
            accelerator, layer, spatial_mapping, temporal_mapping

    def run(self) -> Generator[Tuple['CostModelEvaluation', Any], None, None]:
        """
        Run the cost model stage by calling the internal zigzag cost model with the correct inputs.
        """
        if isinstance(self.layer, LayerNode):
            self.cme = CostModelEvaluation(accelerator=self.accelerator, layer=self.layer, spatial_mapping=self.spatial_mapping, temporal_mapping=self.temporal_mapping)
            yield (self.cme, None)

        elif isinstance(self.layer, MemNode):
            self.moe = MemoryOperatorEvaluation(accelerator=self.accelerator, layer=self.layer)
            yield (self.moe, None)

    def is_leaf(self) -> bool:
        return True