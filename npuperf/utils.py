import pickle
import os
from pprint import pprint
from copy import deepcopy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from npuperf.classes.cost_model.cost_model import CostModelEvaluation


def pickle_deepcopy(to_copy):
    copy = None
    copied = False
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:
        pass
        # fallback to other options

    if not copied:
        return deepcopy(to_copy)


def export(export_path, file_to_export):
    """export dict format file at export_path
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, 'a') as ff:
        pprint(object=file_to_export, stream=ff, indent=4, width=150, sort_dicts=False)
    print(f'export at path {export_path} DONE!')