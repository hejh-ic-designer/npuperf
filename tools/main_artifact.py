from npuperf.classes.stages import *
import numpy as np
import pickle
import logging as _logging
from plot_artifact import plot_Fig12_total_en_and_la_heatmap

#log config
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

####### WHERE THE RESULT FILES WILL BE SAVED TO (USERS CAN CHANGE) #######
result_saving_path = './result_pickle_files'
##########################################################################

# actual run settings:
# parse dfmode: 1 for fully recompute, 2 for H cached V recompute, 3 for fully cached
df_modes = ((False, False), (True, False), (True, True)) #: true: cache, false: re-compute, so df_modes = ((fully re-compute),(H-cache V-recom),(fully cache))
df_x_tilesizes = (1, 4, 16, 60, 240, 960)
df_y_tilesizes = (1, 4, 18, 72, 270, 540)
plotinfo = np.random.rand(3, 2, 6, 6)

class CS1_Result_Collector_Stage(Stage):
    """
    Collects the info required to the plot into the global plotinfo variable
    """

    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)

    def run(self):
        """
        Runs this stage
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        for cme, extra_info in substage.run():
            pass
            i0 = df_modes.index((self.kwargs['df_horizontal_caching'], self.kwargs['df_vertical_caching']))
            i2 = df_y_tilesizes.index(self.kwargs['df_tilesize_y'])
            i3 = df_x_tilesizes.index(self.kwargs['df_tilesize_x'])
            plotinfo[i0, 0, i2, i3] += cme.energy_total
            plotinfo[i0, 1, i2, i3] += cme.latency_total1
        return  # these two line makes this a generator, as required per definition of Stage,
        # although an empty one (intended).
        # We don't care about the results anymore, the relevant metrics are already gathered by this stage
        yield None, None


mainstage = MainStage([
    WorkloadAndAcceleratorParserStage,  # 解析负载和加速器
    GeneralParameterIteratorStage,      # 参数迭代
    CS1_Result_Collector_Stage,         # Case 1的结果收集
    DumpStage,                          # 转存到pkl
    DfStackCutIfWeightsOverflowStage,   # layer stacking, 若权重数据过多溢出，则进行stack 划分
    DepthFirstStage_ref,                # DF
    SpatialMappingConversionStage,      # 空间映射
    RemoveExtraInfoStage,               # 移除多余信息节省内存
    MinimalEnergyStage,                 # 寻找能量最优点
    LomaStage,                          # LOMA 时间映射搜索
    ZigZagCostModelStage_ref            # Cost Model 能量&延时估计
],
    loma_lpf_limit = 8,
    # loop prime factors. the lower it is, the faster propgram runs.
    # running time is 18hr for value 8 and 45min for value 6.
    workload_path='inputs.WL.Meta_prototype.workload_fsrcnn',
    accelerator_path='inputs.HW.Meta_prototype_DF',
    result_saving_path=result_saving_path,

    dump_filename_pattern = '{result_saving_path}/{df_tilesize_x}_{df_tilesize_y}_{df_horizontal_caching}_{df_vertical_caching}.pkl',
    general_parameter_iterations = {('df_horizontal_caching', 'df_vertical_caching'): df_modes,
                                  'df_tilesize_x': df_x_tilesizes,
                                  'df_tilesize_y': df_y_tilesizes}
)

if __name__ == '__main__':
    mainstage.run()
    with open(f'{result_saving_path}/plotinfo.pickle', 'wb') as f:
        pickle.dump(plotinfo, f, -1)
    plot_Fig12_total_en_and_la_heatmap(plotinfo, block=True)
