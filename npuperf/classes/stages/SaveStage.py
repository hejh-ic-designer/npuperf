import json
import logging
import os
import pickle
from typing import Any, Callable, Generator, List, Tuple

import numpy as np

from npuperf.classes.cost_model.cost_model import CostModelEvaluation
from npuperf.classes.cost_model.mem_engine import MemoryOperatorEvaluation
from npuperf.classes.stages.Stage import Stage
from npuperf.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


class CompleteSaveStage(Stage):
    """
    Class that passes through all results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the complete save stage by running the substage and saving the CostModelEvaluation json representation.
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation | MemoryOperatorEvaluation
            if type(cme.layer) == list:
                filename = self.dump_filename_pattern.replace("?", "overall_complete")
            else:
                filename = self.dump_filename_pattern.replace("?", f"{cme.layer}")
            self.save_to_json(cme, filename=filename)
            logger.info(f"Saved {cme} with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}")
            yield cme, extra_info

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, "__jsonrepr__"):
            return obj.__jsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __jsonrepr__ method.")


class SimpleSaveStage(Stage):
    """
    Class that passes through results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    In this simple version, only the energy total and latency total are saved.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation | MemoryOperatorEvaluation
            if type(cme.layer) == list:
                filename = self.dump_filename_pattern.replace("?", "overall_simple")
            else:
                filename = self.dump_filename_pattern.replace("?", f"{cme.layer}_simple")
            self.save_to_json(cme, filename=filename)
            logger.info(f"Saved {cme} with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}")
            yield cme, extra_info

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, "__simplejsonrepr__"):
            return obj.__simplejsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __simplejsonrepr__ method.")


class SumAndSaveAllLayersStage(Stage):
    """
    Class that passes through all results yielded by substages, but saves the sum of results as a json list to a file
    at the end of the iteration.
    In this version, only the sum of energy total and sum of latency total are saved.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    class Cme_or_Moe:

        def __init__(self, en, la, mem_en=0, mac_en=0, act_en=0, mac_la=0, onload_la=0, offload_la=0, ideal_com=0, spa_stl=0, tem_stl=0):
            self.en = en
            self.la = la
            self.mem_en = mem_en
            self.mac_en = mac_en
            self.act_en = act_en
            self.mac_la = mac_la
            self.onload_la = onload_la
            self.offload_la = offload_la
            self.ideal_com = ideal_com
            self.spa_stl = spa_stl
            self.tem_stl = tem_stl

        def __add__(self, other):
            sum = pickle_deepcopy(self)
            sum.en += other.en
            sum.la += other.la
            sum.mem_en += other.mem_en
            sum.mac_en += other.mac_en
            sum.act_en += other.act_en
            sum.mac_la += other.mac_la
            sum.onload_la += other.onload_la
            sum.offload_la += other.offload_la
            sum.ideal_com += other.ideal_com
            sum.spa_stl += other.spa_stl
            sum.tem_stl += other.tem_stl
            return sum

        def __simplejsonrepr__(self):
            """
            Simple JSON representation used for saving this object to a simple json file.
            """
            return {
                "energy(mJ)": self.en / 1e9,
                "latency(mC)": self.la / 1e6,
                "energy breakdown":{
                    "memory energy": self.mem_en / 1e9,
                    "mac energy": self.mac_en / 1e9,
                    "activation energy": self.act_en / 1e9,
                },
                "latency breakdown":{
                    "mac latency": self.mac_la / 1e6,
                    "onloading latency": self.onload_la / 1e6,
                    "offloading latency": self.offload_la / 1e6,
                    "ideal computation latency": self.ideal_com / 1e6,
                    "spatial stall": self.spa_stl / 1e6,
                    "temporal stall": self.tem_stl / 1e6,
                },
            }

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        先把 cme 和 moe 的 en 和 la 数据存在 Cme_or_Moe 中, 然后对 Cme_or_Moe 作累加, 完成累加后保存在json 文件
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        filename = self.dump_filename_pattern.replace("?", "All_Layers")

        sum_cme_or_moe = self.Cme_or_Moe(en=0, la=0)
        all_cmes = []

        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation | MemoryOperatorEvaluation
            all_cmes.append(cme)
            if isinstance(cme, CostModelEvaluation):
                cme_or_moe = self.Cme_or_Moe(en=cme.energy_total,
                                             la=cme.latency_total2,
                                             mem_en=cme.mem_energy,
                                             mac_en=cme.MAC_energy,
                                             act_en=cme.activation_energy,
                                             mac_la=cme.latency_total0,
                                             onload_la=cme.data_loading_cycle,
                                             offload_la=cme.data_offloading_cycle,
                                             ideal_com=cme.ideal_cycle,
                                             spa_stl=cme.spatial_stall_cycle,
                                             tem_stl=cme.SS_comb)
            elif isinstance(cme, MemoryOperatorEvaluation):
                cme_or_moe = self.Cme_or_Moe(en=cme.extra_data_copy_en, la=cme.extra_data_copy_la)
            sum_cme_or_moe += cme_or_moe

        self.save_to_json(sum_cme_or_moe, filename=filename)
        logger.info(f"Saved sum of all layers with energy {sum_cme_or_moe.en:.3e} and latency {sum_cme_or_moe.la:.3e} to {filename}")
        yield sum_cme_or_moe, all_cmes

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, "__simplejsonrepr__"):
            return obj.__simplejsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __simplejsonrepr__ method.")


class PickleSaveStage(Stage):
    """
    Class that dumps all received CMEs into a list and saves that list to a pickle file.
    """

    def __init__(self, list_of_callables, *, pickle_filename, **kwargs):
        """
        :param list_of_callables: see Stage
        :param pickle_filename: output pickle filename
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.pickle_filename = pickle_filename

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        This should be placed above a ReduceStage such as the SumStage, as we assume the list of CMEs is passed as extra_info
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        # for id, (cme, extra_info) in enumerate(substage.run()):
        #     all_cmes = [cme for (cme, extra) in extra_info]
        #     yield cme, extra_info
        all_cmes = []
        for cme, extra_info in substage.run():
            all_cmes.append(cme)
            yield cme, extra_info
        # After we have received all the CMEs, save them to the specified output location.
        dirname = os.path.dirname(self.pickle_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(self.pickle_filename, "wb") as handle:
            pickle.dump(all_cmes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved pickled list of {len(all_cmes)} CMEs to {self.pickle_filename}.")
