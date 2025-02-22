"""
从pickle 文件中读取 List[CME], 作图, 统计每层的energy & latency
"""
import pickle
import os
from npuperf.classes.cost_model.cost_model import CostModelEvaluation
from npuperf.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown, bar_plot_cost_model_evaluations_total

experiment_name = 'Meta_prototype--mobile_vit'
pickle_path = f'outputs/{experiment_name}/{experiment_name}.pickle'  #! 路径注意区分 stationary
png_bd_path = f"result_plot/{experiment_name}/breakdown.png"    #! 路径注意区分 stationary
png_tt_path = f"result_plot/{experiment_name}/total.png"    #! 路径注意区分 stationary

with open(pickle_path, "rb") as handle:
    list_of_cme = pickle.load(handle)
cmes = [cme_obj for cme_obj in list_of_cme if isinstance(cme_obj, CostModelEvaluation)]

print(f'Charting plot, please wait ...')
os.makedirs(os.path.dirname(png_bd_path), exist_ok=True)
os.makedirs(os.path.dirname(png_tt_path), exist_ok=True)
bar_plot_cost_model_evaluations_breakdown(cmes, png_bd_path)
bar_plot_cost_model_evaluations_total(cmes, png_tt_path)
print(f'Done!, png file at {png_bd_path} and {png_tt_path}')


