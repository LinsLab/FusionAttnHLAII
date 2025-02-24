import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle



def sub_dataset_graph_more(ax, data, mg_hla_data):
    methods = ['comblib', 'Consensus3', 'SMM', 'Netmhcpan_el', 'Netmhcpan_el-4.2', 'Netmhcpan_ba', 'Netmhcpan_ba-4.2', 'HLAIImaster']
    metrics = ['AUC', 'ACC', 'MCC', 'F1']
    
    gap = 0.013  # 方法之间没有间隙
    color_palette = sns.color_palette("Purples", n_colors=len(methods))
    method_colors = {method: color for method, color in zip(methods, color_palette)}

    # 为每个性能指标设置基础位置
    x = np.arange(len(metrics)) * 1.5  # 增加间隔以清晰展示

    # 绘制每种方法的性能指标条形图
    bar_width = 0.13
    for i, method in enumerate(methods):
        ax.bar(x + i * (bar_width + gap), data[i], bar_width, label=method, color=method_colors[method], edgecolor='black', alpha=0.7)
        if i == 0:
            ax.bar(x + i * (bar_width + gap), mg_hla_data[i] - data[i], bar_width,
                   bottom=data[i], facecolor='none', edgecolor='black', linestyle='--', alpha=0.7, capstyle='round')
        else:
            ax.bar(x + i * (bar_width + gap), mg_hla_data[i] - data[i], bar_width,
                   bottom=data[i], facecolor='none', edgecolor='black', linestyle='--', alpha=0.7, capstyle='round')
    
    # 绘制虚线分隔不同的性能指标
    for i in range(1, len(metrics)):
        ax.axvline(x[i] - 3 * bar_width, color='gray', linestyle='--', alpha=0.7)

    # 设置坐标轴和标题
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value', fontsize=18)
    ax.set_ylim(25, 100)
    ax.set_title('Test set1 Metrics', fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # **Step 1: 获取自动生成的第一个图例**
    handles1, labels1 = ax.get_legend_handles_labels()

    # **Step 2: 创建 FusionHLAII 的单独图例**
    fusion_legend = [Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='none', linestyle='--', linewidth=1.5, label='FusionHLAII')]

    # **Step 3: 在左上角显示单独的 FusionAttnHLAII 图例**
    second_legend = ax.legend(handles=fusion_legend, loc='upper right', fontsize=12, frameon=False)
    ax.add_artist(second_legend)
    
    plt.savefig('../figure/Independent-subset Metrics.png',bbox_inches="tight")
    plt.savefig('../figure/Independent-subset Metrics.pdf',bbox_inches="tight")
    plt.savefig('../figure/Independent-subset Metrics.svg',bbox_inches="tight")

    # **返回第一个图例供外部使用**
    return handles1, labels1

def plot_performance_combined(ax, results):
    # 定义基线方法及对应的 marker
    baseline_methods = [
        "NetMHCIIpan-EL-4.2",
        "NetMHCIIpan-BA-4.2",
        "NetMHCIIpan-EL",
        "NetMHCIIpan-BA",
        "Consensus3",
        "Comblib",
        "smm",
        "HLAIImaster"
    ]
    marker_map = {
        "NetMHCIIpan-EL-4.2": 'o',   # 圆形
        "NetMHCIIpan-BA-4.2": '^',   # 上三角形
        "NetMHCIIpan-EL":    'v',    # 下三角形
        "NetMHCIIpan-BA":    'D',    # 菱形
        "Consensus3":        's',    # 方形
        "Comblib":           'p',    # 五边形
        "smm":               '*',    # 星形
        "HLAIImaster":       'h'     # 六边形
    }

    metrics = ["AUC", "ACC", "MCC", "F1"]
    metric_centers = np.arange(len(metrics)) * 3
    section_width = 3
    margin = 1.2
    offset = 0.1

    for i, metric in enumerate(metrics):
        center = metric_centers[i]
        method_x_positions = np.linspace(center - margin, center + margin, len(baseline_methods))
        for j, method in enumerate(baseline_methods):
            baseline_val = results.get(method, {}).get(metric, None)
            fusion_key = "FusionHLAII_" + method
            fusion_val = results.get(fusion_key, {}).get(metric, None)
            
            if baseline_val is not None:
                ax.plot(method_x_positions[j], baseline_val, marker=marker_map[method],
                        color='black', markersize=8, markerfacecolor='none', markeredgewidth=2,
                        label=method if i == 0 else "")  # 只在第一个指标中添加标签
            if fusion_val is not None:
                ax.plot(method_x_positions[j], fusion_val, marker=marker_map[method],
                        color='red', markersize=8, markerfacecolor='none', markeredgewidth=2)  # 只在第一个指标中添加标签

    for i in range(1, len(metrics)):
        ax.axvline(x=metric_centers[i] - section_width / 2, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xticks(metric_centers)
    ax.set_xticklabels(metrics, fontsize=20)
    ax.set_ylabel('Performance', fontsize=20)
    ax.set_title('Test set2 Metrics', fontsize=20)
    ax.set_ylim(10, 100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # **获取第一个图例的 handles 和 labels**
    handles1, labels1 = ax.get_legend_handles_labels()
    legend_elements = [Line2D([0], [0], marker=marker_map[method], color='black', markersize=8, 
                              markerfacecolor='none', markeredgewidth=2, linestyle='None', label=method) 
                       for method in baseline_methods]

    # **创建红色和黑色点的第二个图例**
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', label='Baseline Methods', markerfacecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='FusionHLAII', markerfacecolor='red', markersize=12)
    ]

    # **使用 `add_artist` 来显示第二个图例，不覆盖第一个图例**
    second_legend = ax.legend(handles=custom_lines, loc='upper right', fontsize=12)
    ax.add_artist(second_legend)
    
    # 保存图像（根据需要修改保存路径）
    plt.savefig('../figure/external_Metrics.png', bbox_inches="tight")
    plt.savefig('../figure/external_Metrics.pdf', bbox_inches="tight")
    plt.savefig('../figure/external_Metrics.svg', bbox_inches="tight")

    # **返回第一个图例的 handles 和 labels 供外部使用**
    return handles1, labels1,legend_elements


    
if __name__ == '__main__':
    '''
    mapping = select_colors_for_methods_seaborn()
    print(mapping)
    
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    #对independent的全部数据方法作图
    performance_data = np.array([
        
        [0.9552, 0.802, 0.6496, 0.7579],   # Netmhcpan_ba
        [0.9568, 0.7952, 0.6433, 0.7449],   # Netmhcpan_el
        [0.9162, 0.7901, 0.6222, 0.7445],   # netmhcstabpan
        [0.93601, 0.93602, 0.8721, 0.9364], # HLAB
        [0.9778, 0.9294, 0.8588, 0.9298] ,# TransPHLA
        [0.9867,0.9466,0.8931,0.9468]
        
    ])
    #all_data_graph(performance_data)
    all_data_graph(ax0,performance_data)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mapping = select_colors_for_methods_seaborn()
    print(mapping)

    data = np.array([
        [93.47, 78.12, 61.54, 72.63],  # Ann
        [93.4, 70.33, 49.46, 58.66],   # Pickpocket
        [93.21, 78.63, 62.39, 73.41],  # Consensus
        [91.05, 78.99, 60.74, 75.38],  # SMM
        [91.37, 78.9, 60.8, 75.38],     # SMMPMBEC
        [94.77, 78.99, 63.12, 73.88],#ACME
        [97.69, 91.39, 83.32, 90.88]   # Anthem
    ])

    mg_hla_data = np.array([
        [98.48, 94.21, 88.41, 94.23],  # MGHLA-ANN
        [98.80, 95.10, 90.20, 95.12],  # MGHLA-Pickpocket
        [98.48, 94.21, 88.42, 94.23],  # MGHLA-Consensus
        [98.65, 94.71, 89.42, 94.74],  # MGHLA-SMM
        [98.65, 94.72, 89.43, 94.75],   # MGHLA-SMMPMBEC
        [98.62, 94.50, 88.99, 94.52],
        [98.67,94.66,89.33,94.69] #MGHLA_Anthem
    ])
    
    

    # 调用绘图函数
    sub_dataset_graph_more(ax1,data,mg_hla_data)

    
    # 创建一个空白的axes仅用于显示图例
    
    # 提取图例
    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 =ax1.get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1
    # 创建一个新的空白图形
    ncol = len(handles) // 2
    fig_leg = plt.figure(figsize=(10, 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.legend(handles, labels, loc='center', ncol=ncol,frameon=False)
    ax_leg.axis('off')  # 隐藏坐标轴

    # 保存图例
    fig_leg.savefig('legend1.pdf', dpi=300, bbox_inches='tight', transparent=True)
    fig_leg.savefig('legend1.svg', dpi=300, bbox_inches='tight', transparent=True)
    '''
    
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    

    data = np.array([
        [68.77, 64.46, 28.90, 67.95],  # comblib
        [79.61, 62.56, 32.78, 46.85],   # Consensus
        [79.66, 72.31, 44.67, 72.26],  # SMM
        [85.53, 63.65, 39.49, 45.34],   # Netmhcpan_el
        [85.52, 63.83, 39.82, 45.72],   # Netmhcpan_el-4.2
        [83.54, 75.64, 51.49, 75.13],   # Netmhcpan_ba
        [85.16, 76.96, 54.07, 76.69], # Netmhcpan_ba-4.2
        [92.04,83.66,67.58,84.38]  #_HLAIImaster
        
    ])

    mg_hla_data = np.array([
        [93.42, 85.54, 71.06, 86.08],  # FusionHLAII-comblib
        [92.54, 84.55, 69.11, 85.00],  # FusionHLAII-Consensus
        [92.18, 84.17, 68.34, 84.62],  # FusionHLAII-SMM
        [95.37, 88.13, 76.25, 88.45],  # FusionHLAII-Netmhcpan_el
        [95.37, 88.13, 76.25, 88.45],   # FusionHLAII-Netmhcpan_el-4.2
        [95.37, 88.13, 76.25, 88.45],  #FusionHLAII-Netmhcpan_ba
        [95.37, 88.13, 76.25, 88.45], #FusionHLAII_Netmhcpan_ba-4.2
        [95.37, 88.13, 76.25, 88.45] #FusionHLAII_HLAIImaster
    ])
    
    

    # 调用绘图函数
    #sub_dataset_graph_more(ax1,data,mg_hla_data)
    handles1, labels1 = sub_dataset_graph_more(ax1, data, mg_hla_data)

    
    # 创建一个空白的axes仅用于显示图例
    
    # 提取图例
    #handles0, labels0 = ax0.get_legend_handles_labels()
    
    
    #handles1, labels1 =ax1.get_legend_handles_labels()
    
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    results = {
    # 基线方法结果（注意这里仅给出示例数值，实际请替换为你的结果）
    "NetMHCIIpan-EL-4.2": {"AUC": 79.80, "ACC":63.22, "MCC": 40.00, "F1": 52.14},
    "NetMHCIIpan-BA-4.2": {"AUC": 72.70, "ACC":67.01, "MCC": 34.60, "F1": 68.20},
    "NetMHCIIpan-EL":    {"AUC": 80.80, "ACC":63.18, "MCC": 39.75, "F1": 52.19},
    "NetMHCIIpan-BA":    {"AUC": 72.70, "ACC":66.47, "MCC": 33.71, "F1": 67.45},
    "Consensus3":        {"AUC": 68.45, "ACC": 57.03, "MCC": 23.13, "F1": 41.81},
    "Comblib":           {"AUC": 59.12, "ACC": 56.12, "MCC": 12.23, "F1": 61.01},
    "smm":               {"AUC": 67.69, "ACC": 62.77, "MCC": 26.02, "F1": 63.00},
    "HLAIImaster":       {"AUC": 73.77, "ACC": 66.63, "MCC":32.12, "F1": 70.7},

    # 对应FusionHLAII在各基线方法可预测子集上的结果，键名格式为"FusionAttnHLAII_基线方法名"
    "FusionHLAII_NetMHCIIpan-EL-4.2": {"AUC": 77.15, "ACC": 70.20, "MCC": 40.90, "F1": 71.43},
    "FusionHLAII_NetMHCIIpan-BA-4.2": {"AUC": 77.15, "ACC": 70.20, "MCC": 40.90, "F1": 71.43},
    "FusionHLAII_NetMHCIIpan-EL":    {"AUC": 77.15, "ACC": 70.20, "MCC": 40.90, "F1": 71.43},
    "FusionHLAII_NetMHCIIpan-BA":    {"AUC": 77.15, "ACC": 70.20, "MCC": 40.90, "F1": 71.43},
    "FusionHLAII_Consensus3":        {"AUC": 77.83, "ACC": 70.69, "MCC": 41.85, "F1": 71.13},
    "FusionHLAII_Comblib":           {"AUC": 76.30, "ACC": 69.77, "MCC": 39.52, "F1":69.01},
    "FusionHLAII_smm":       {"AUC": 78.15, "ACC": 71.12, "MCC": 42.68, "F1": 71.48},
    "FusionHLAII_HLAIImaster":       {"AUC": 77.15, "ACC": 70.20, "MCC": 40.90, "F1": 71.43},
    }
    
    handles2, labels2,legend_elements=plot_performance_combined(ax2,results)
    #handles2, labels2 =ax2.get_legend_handles_labels()
    #legend_elements=plot_performance_combined(ax2,results)
    
    
    
    
    handles = handles1 + legend_elements
    labels = labels1 + labels2
    # 创建一个新的空白图形
    ncol = len(handles) // 2
    #ncol = len(handles1)//2
    fig_leg = plt.figure(figsize=(15, 2))
    ax_leg = fig_leg.add_subplot(111)
    #ax_leg.legend(handles, labels, loc='center', ncol=ncol,frameon=False)
    ax_leg.legend(handles, labels, loc='center', ncol=ncol, frameon=False, 
              fontsize=20,  # 增大字体
              labelspacing=1.1)  # 增大行之间的距离
    #ax_leg.legend(handles=handles1 +legend_elements, loc='center', ncol=ncol, frameon=False)
    ax_leg.axis('off')  # 隐藏坐标轴

    # 保存图例
    fig_leg.savefig('../figure/legend1.pdf', dpi=300, bbox_inches='tight', transparent=True)
    fig_leg.savefig('../figure/legend1.png', dpi=300, bbox_inches='tight', transparent=True)
    #fig_leg.savefig('legend1.svg', dpi=300, bbox_inches='tight', transparent=True)

    
    
    
    
    