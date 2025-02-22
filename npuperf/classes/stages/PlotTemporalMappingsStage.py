import os
import logging
from typing import Generator, Any, Tuple
from npuperf.classes.stages.Stage import Stage
from npuperf.classes.cost_model.cost_model import CostModelEvaluation
from npuperf.visualization.results.print_mapping_tofile import print_mapping_tofile
from npuperf.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown, bar_plot_cost_model_evaluations_total
from npuperf.visualization.graph.memory_hierarchy_word_access import visualize_memory_hierarchy_graph_Word_Access
logger = logging.getLogger(__name__)

## Class that passes through all results yielded by substages, but keeps the TMs cme's and saves a plot.
class PlotTemporalMappingsStage(Stage):

    ## The class constructor
    # @param list_of_callables: see Stage
    # @param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
    # in kwargs (thus supplied by higher level runnables)
    # @param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
    def __init__(self, list_of_callables, *, plot_filename_pattern, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.plot_filename_pattern = plot_filename_pattern

    ## Run the compare stage by comparing a new cost model output with the current best found result.
    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        self.cmes = []
        for cme, extra_info in substage.run():
            self.cmes.append(cme)
            yield cme, extra_info

        logger.info(f'Charting all result information in directory {self.plot_filename_pattern.replace("?.png", "")}, Please Wait......')
        # 先在 self.cmes list 里把 moe 挑出来，因为这个暂时还画不了
        self.cmes = [cme_obj for cme_obj in self.cmes if isinstance(cme_obj, CostModelEvaluation)]
        self.plot_breakdown()
        self.plot_total()
        self.print_temporal_mapping()
        # self.plot_memhier_word_access()   # 这个是每层都画一张mem word access的图，网络层过多则绘图太多，且这是不必要的信息

    def plot_breakdown(self):
        # plot en & la Breakdown
        filename_bd = self.plot_filename_pattern.replace("?", "breakdown")
        os.makedirs(os.path.dirname(filename_bd), exist_ok=True)
        bar_plot_cost_model_evaluations_breakdown(self.cmes, filename_bd)
        logger.info(f'plot CME breakdown at path {filename_bd}')

    def plot_total(self):
        # plot en & la Total
        filename_tt = self.plot_filename_pattern.replace("?", "total")
        os.makedirs(os.path.dirname(filename_tt), exist_ok=True)
        bar_plot_cost_model_evaluations_total(self.cmes, filename_tt)
        logger.info(f'plot CME total at path {filename_tt}')

    def print_temporal_mapping(self):
        # print temporal mapping in good format
        filename_tm = self.plot_filename_pattern.replace("?.png", "temoporal_mapping.txt")
        print_mapping_tofile(self.cmes, file_path=filename_tm)
        logger.info(f'print temporal mapping at path {filename_tm}')

    def plot_memhier_word_access(self):
        # plot memhier word access
        filename_wa = self.plot_filename_pattern.replace("?.png", "memhier_word_access/")   # f'result_plot/{experiment_id}/
        os.makedirs(os.path.dirname(filename_wa), exist_ok=True)
        for id, cme in enumerate(self.cmes):
            mem_hier = cme.accelerator.get_core(cme.core_id).get_memory_hierarchy()
            mem_word_access = cme.memory_word_access
            save_path = f'{filename_wa}{cme.layer}.png'
            if cme.layer.TYPE != 'Add':
                visualize_memory_hierarchy_graph_Word_Access(mem_hier, mem_word_access, save_path)
        logger.info(f'plot memory hierarchy word access graph at path {filename_wa}')


