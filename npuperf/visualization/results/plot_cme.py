from typing import Dict, List, Tuple
from typing import TYPE_CHECKING
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from npuperf.classes.mapping.combined_mapping import FourWayDataMoving
from npuperf.classes.cost_model.cost_model import CostModelEvaluation

# MPL FONT SIZES
SMALLEST_SIZE = 8
SMALLER_SIZE = 10
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 20
plt.rc("font", size=SMALLEST_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def bar_plot_cost_model_evaluations_total(
    cmes: List[CostModelEvaluation],
    save_path: str = "plot.png",
):
    """Plot total energy and latency of each cost model evaluation in a bar chart.

    Args:
        cmes (List[CostModelEvaluation]): List of CostModelEvaluations to compare.
        save_path (str): Path to save the plot to.
    """
    # labels = list(range(len(cmes)))
    # labels = [f'{i}_{cme.layer.TYPE}' for (i, cme) in enumerate(cmes)]
    labels = [f'{cme.layer.id}_{cme.layer.TYPE}' for cme in cmes]

    energies = [cme.energy_total for cme in cmes]
    latencies = [cme.latency_total2 for cme in cmes]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(70, 20))
    ax2 = ax1.twinx()

    colormap = plt.get_cmap("Set1")
    color_energy = colormap.colors[0]
    color_latency = colormap.colors[1]

    h1 = rects1 = ax1.bar(
        x - width / 2, energies, width, label="Energy", color=color_energy
    )
    h2 = rects2 = ax2.bar(
        x + width / 2, latencies, width, label="Latency", color=color_latency
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel("Energy [pJ]", fontsize=15)
    ax2.set_ylabel("Latency [cycle]", fontsize=15)
    ax1.set_xticks(x, labels, rotation=45)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        bbox_to_anchor=(0.5, 1.035),
        loc="lower center",
        borderaxespad=0,
        ncol=2,
    )

    ax1.bar_label(rects1, padding=3, fmt="%.1e")
    ax2.bar_label(rects2, padding=3, fmt="%.1e")
    # ax1.figure.texts.append(ax1.texts.pop())
    # ax2.figure.texts.append(ax2.texts.pop())

    # ax1.set_title(fig_title)
    fig.tight_layout()
    plt.savefig(save_path)


def bar_plot_cost_model_evaluations_breakdown(
    cmes: List[CostModelEvaluation], save_path: str, xtick_rotation=90
):
    memory_word_access_summed = {
        d: defaultdict(lambda: defaultdict(lambda: FourWayDataMoving(0, 0, 0, 0)))
        for d in range(len(cmes))
    }
    mac_costs = defaultdict(lambda: 0)
    memory_instances = {}
    la_break_down = {
        d: {
            "Ideal computation": 0,
            "Spatial stall": 0,
            "Temporal stall": 0,
            "Data loading": 0,
            "Data off-loading": 0,
        }
        for d in range(len(cmes))
    }
    la_tot = {d: 0 for d in range(len(cmes))}

    for d, cme in enumerate(cmes):
        mh = cme.accelerator.get_core(cme.layer.core_allocation).memory_hierarchy
        mac_costs[d] = cme.MAC_energy
        la_break_down[d]["Ideal computation"] = cme.ideal_cycle
        la_break_down[d]["Spatial stall"] = cme.ideal_temporal_cycle - cme.ideal_cycle
        la_break_down[d]["Temporal stall"] = (
            cme.latency_total0 - cme.ideal_temporal_cycle
        )
        la_break_down[d]["Data loading"] = cme.latency_total1 - cme.latency_total0
        la_break_down[d]["Data off-loading"] = cme.latency_total2 - cme.latency_total1
        la_tot[d] = cme.latency_total2
        for operand in cme.energy_breakdown_further:
            mem_op = cme.layer.memory_operand_links[operand]
            operand_memory_levels = mh.get_memory_levels(mem_op)
            for j in range(len(cme.energy_breakdown_further[operand])):
                mem = operand_memory_levels[j].name
                memory_instances[mem] = operand_memory_levels[j]
                memory_word_access_summed[d][operand][mem] += cme.energy_breakdown_further[operand][j]

    all_mems = set()
    for v in memory_word_access_summed.values():
        for vv in v.values():
            for vvv in vv.keys():
                all_mems.add(vvv)
    all_mems = sorted(
        list(all_mems), key=lambda m: memory_instances[m].memory_instance.size
    )
    all_ops = set()
    for v in memory_word_access_summed.values():
        for vv in v.keys():
            all_ops.add(vv)
    all_ops = sorted(list(all_ops))

    """ plotting start """
    """ Energy part """

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(70, 32))   # 原本是 (35, 16)
    hues = np.linspace(0, 1, len(all_ops) + 1)[:-1]
    hatches = ["////", "\\\\\\\\", "xxxx", "++++"]
    x = 0
    xticks = {}
    for d, cme in enumerate(cmes):
        total_energy = 0
        startx_of_layer = x
        # mac
        ax1.bar([x], [mac_costs[d]], width=1, bottom=0, facecolor="k")
        total_energy += mac_costs[d]
        highest_bar = mac_costs[d]
        xticks[x] = "MAC"
        x += 1
        # mems
        for mem in all_mems:
            bottom = 0
            for op_i, operand in enumerate(all_ops):
                for dir_i, dir in enumerate(memory_word_access_summed[d][operand][mem]):
                    height = memory_word_access_summed[d][operand][mem][dir]
                    ax1.bar(
                        [x],
                        [height],
                        width=1,
                        bottom=[bottom],
                        facecolor=hsv_to_rgb((hues[op_i], 1, 1)),
                        hatch=hatches[dir_i],
                    )

                    bottom += height
            xticks[x] = mem
            total_energy += bottom
            x += 1
            highest_bar = max(bottom, highest_bar)
        x
        ax1.text(
            x * 0.5 + startx_of_layer * 0.5,
            1.05 * highest_bar,
            "tot:{:,d}".format(int(total_energy)),
            horizontalalignment="center",
            verticalalignment="bottom",
            weight="bold",
        )
        x += len(all_mems) / 4

    for op, h in zip(all_ops, hues):
        ax1.bar(0, 0, width=1, facecolor=hsv_to_rgb((h, 1, 1)), label=op)

    for dir_i, dir in enumerate(memory_word_access_summed[d][operand][mem]):
        ax1.bar(
            [0],
            [0],
            width=1,
            bottom=0,
            facecolor=(1, 1, 1),
            hatch=hatches[dir_i],
            label=dir,
        )

    ax1.legend(loc="upper left")
    ax1.set_xticks(list(xticks.keys()), list(xticks.values()), rotation=xtick_rotation)
    ax1.set_ylim(0, 1.1 * ax1.get_ylim()[1])

    ax1.set_ylabel("Energy (pJ)", fontsize=15)

    """ Latency part """
    x2 = list(range(len(la_break_down)))
    la_labels = [f'{cme.layer.id}_{cme.layer.TYPE}' for cme in cmes]
    # la_labels = [f'{i}_{cme.layer.TYPE}' for (i, cme) in enumerate(cmes)]       # e.g.: '5_Conv'

    y2 = {ky: [] for ky in la_break_down[0].keys()}
    for _, design_point in la_break_down.items():
        for ky, val in design_point.items():
            y2[ky].append(val)

    hues = np.linspace(0, 1, len(y2) + 1)[:-1]

    for idx, (ky, va) in enumerate(y2.items()):
        if idx == 0:
            ax2.bar(
                np.array(x2),
                va,
                width=0.4,
                color=hsv_to_rgb((hues[idx], 1, 1)),
                label=ky,
            )
            li_pre = va
        else:
            ax2.bar(
                np.array(x2),
                va,
                width=0.4,
                color=hsv_to_rgb((hues[idx], 1, 1)),
                label=ky,
                bottom=li_pre,
            )
            li_pre = [x + y for x, y in zip(li_pre, va)]

    for x in x2:
        ax2.text(
            x,
            la_tot[x] * 1.05,
            "tot:{:,d}".format(int(la_tot[x])),
            horizontalalignment="center",
            verticalalignment="bottom",
            weight="bold",
        )
    ax2.legend()
    # ax2.set_xticks(x2, x2, rotation=xtick_rotation)
    ax2.set_xticks(x2, la_labels, rotation=xtick_rotation/2)      # x 坐标轴标签设置, 旋转45度
    ax2.set_ylim(0, 1.1 * ax2.get_ylim()[1])
    ax2.set_xlabel("Layers", fontsize=15)
    ax2.set_ylabel("Latency (cycle)", fontsize=15)

    fig.tight_layout()
    plt.savefig(save_path)

def bar_plot_cost_model_evaluations_total_ref(
    cmes: List[CostModelEvaluation],
    labels,
    save_path: str = "plot.png",
):
    """Plot total energy and latency of each cost model evaluation in a bar chart.

    Args:
        cmes (List[CostModelEvaluation]): List of CostModelEvaluations to compare.
        save_path (str): Path to save the plot to.
    """
    assert len(cmes) == len(
        labels
    ), "Please match a label for each cost model evaluation."
    energies = [cme.energy_total for cme in cmes]
    latencies = [cme.latency_total2 for cme in cmes]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    colormap = plt.get_cmap("Set1")
    color_energy = colormap.colors[0]
    color_latency = colormap.colors[1]

    h1 = rects1 = ax1.bar(
        x - width / 2, energies, width, label="Energy", color=color_energy
    )
    h2 = rects2 = ax2.bar(
        x + width / 2, latencies, width, label="Latency", color=color_latency
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel("Energy [pJ]", fontsize=15)
    ax2.set_ylabel("Latency [cycle]", fontsize=15)
    ax1.set_xticks(x, labels, rotation=45)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        bbox_to_anchor=(0.5, 1.035),
        loc="lower center",
        borderaxespad=0,
        ncol=2,
    )

    ax1.bar_label(rects1, padding=3, fmt="%.1e")
    ax2.bar_label(rects2, padding=3, fmt="%.1e")
    # ax1.figure.texts.append(ax1.texts.pop())
    # ax2.figure.texts.append(ax2.texts.pop())

    # ax1.set_title(fig_title)
    fig.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    import pickle

    with open("../list_of_cmes.pickle", "rb") as handle:
        list_of_cme = pickle.load(handle)
    bar_plot_cost_model_evaluations_breakdown(list_of_cme, "plot.png")
