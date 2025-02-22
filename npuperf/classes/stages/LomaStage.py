from npuperf.classes.opt.temporal.loma.engine import LomaEngine
from npuperf.classes.opt.temporal.loma.engine_stationary import LomaEngine_Stationary
from typing import Generator, Callable, List, Tuple, Any
from npuperf.classes.stages.Stage import Stage


class LomaStage(Stage):
    """
    Class that iterates through the different temporal mappings generated through
    the loop order based memory allocation (loma) engine
    """
    def __init__(self, list_of_callables: List[Callable], *, accelerator, layer, spatial_mapping, **kwargs):
        """
        Note: Initially the engine is set to None.
        When the stage is ran through the run() method, this will be set
        to the loma engine with parameters present in the inputs.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping = accelerator, layer, spatial_mapping
        self.engine = None

    def run(self):
        self.engine = LomaEngine(accelerator=self.accelerator, layer=self.layer, spatial_mapping=self.spatial_mapping,
                                 **self.kwargs)

        for tm in self.engine.run():
            kwargs = self.kwargs.copy()
            kwargs['accelerator'] = self.accelerator
            kwargs['layer'] = self.layer
            kwargs['spatial_mapping'] = self.spatial_mapping
            kwargs['temporal_mapping'] = tm
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (tm, extra_info)


class StationaryLomaStage(Stage):
    """
    在loma engine的基础上限制搜索范围, 使得得到的temporal mapping 符合output stationary, weight stationary, input stationary    \\
    Class that iterates through the different temporal mappings generated through
    the loop order based memory allocation (loma) engine
    """
    def __init__(self, list_of_callables: List[Callable], *, accelerator, layer, spatial_mapping, stationary:str = None, **kwargs):
        """
        Note: Initially the engine is set to None.
        When the stage is ran through the run() method, this will be set
        to the loma engine with parameters present in the inputs.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping = accelerator, layer, spatial_mapping
        self.engine = None
        assert stationary in ['O', 'I', 'W'], "stationary set error! please set stationary to 'O', 'I', or 'W' "
        self.stationary = stationary    # 'O', 'I', 'W'

    def run(self):
        self.engine = LomaEngine_Stationary(accelerator=self.accelerator, layer=self.layer, spatial_mapping=self.spatial_mapping, stationary = self.stationary,
                                 **self.kwargs)

        for tm in self.engine.run():
            kwargs = self.kwargs.copy()
            kwargs['accelerator'] = self.accelerator
            kwargs['layer'] = self.layer
            kwargs['spatial_mapping'] = self.spatial_mapping
            kwargs['temporal_mapping'] = tm
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (tm, extra_info)
