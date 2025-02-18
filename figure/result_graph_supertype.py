import matplotlib.pyplot as plt
import numpy as np
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

def plot_performance(performance):
    """
    根据性能字典绘制条形图。
    
    参数:
        performance (dict): 性能字典，格式如：
            {
                'DP': {'comblib': auc1, 'Consensus': auc2, ...},
                'DQ': {'comblib': auc3, 'Consensus': auc4, ...},
                'DR': {'comblib': auc5, 'Consensus': auc6, ...}
            }
        方法包括：comblib、Consensus、SMM、Netmhcpan_el、Netmhcpan_el-4.2、
                Netmhcpan_ba、Netmhcpan_ba-4.2、HLAIImaster、FusionAttnHLAII.
                
    效果：
        - 横轴为 DP, DQ, DR 三组
        - 纵轴为 AUC 值
        - 每个方法用不同颜色的条形表示，并在不同组之间添加虚线隔断
        - 每个条形添加黑色边框以突出显示
    """
    '''
    # 采用一组差别大但鲜明柔和的颜色（9种）
    color_hex = ['#FF6961',  # Pastel Red
                 '#FFB347',  # Pastel Orange
                 '#FDFD96',  # Pastel Yellow
                 '#77DD77',  # Pastel Green
                 '#AEC6CF',  # Pastel Blue
                 '#CFCFC4',  # Soft Gray
                 '#B39EB5',  # Pastel Purple
                 '#F49AC2',  # Pastel Pink
                 '#CB99C9']  # Pastel Violet

    # 固定方法顺序
    '''
    
    methods = ['comblib', 'Consensus', 'SMM', 'Netmhcpan_ba', 'Netmhcpan_ba-4.2',
               'Netmhcpan_el', 'Netmhcpan_el-4.2', 'HLAIImaster', 'FusionAttnHLAII']
    color_palette = sns.color_palette("Blues", n_colors=len(methods))
    groups = ['DP', 'DQ', 'DR']
    
    method_color_mapping = dict(zip(methods, color_palette))
    num_groups = len(groups)
    num_methods = len(methods)
    
    
    x = np.arange(num_groups)
    bar_width = 0.08
    
    positions = [x - (num_methods / 2) * bar_width + i * bar_width + bar_width/2 
                 for i in range(num_methods)]
    
   
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, method in enumerate(methods):
        
        auc_values = [performance.get(group, {}).get(method, np.nan) for group in groups]
        ax.bar(positions[i], auc_values, width=bar_width, label=method,
               color=method_color_mapping[method], edgecolor='black', linewidth=1)
    
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=18)
    ax.set_ylabel("AUC", fontsize=18)
    ax.set_title("Metrics in DP/DQ/DR", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(60, 100)
    
    
    for i in range(1, num_groups):
        dashed_x = (x[i-1] + x[i]) / 2
        ax.axvline(x=dashed_x, color='gray', linestyle='--')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype.png', bbox_inches="tight")
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype.pdf', bbox_inches="tight")
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype.svg', bbox_inches="tight")
    
    plt.show()
    
    
def plot_hla_prediction_counts(data):
    """
    绘制折线图，展示不同方法预测的HLA分子种类数。
    
    参数:
        data (dict): 嵌套字典，格式如下：
            {
                "DP": {"comblib": count1, "Consensus": count2, ...},
                "DQ": {"comblib": count3, "Consensus": count4, ...},
                "DR": {"comblib": count5, "Consensus": count6, ...}
            }
        其中 count 值为整数，表示各方法对应的预测数量。
    """
    
    methods = ['comblib', 'Consensus3', 'SMM', 'Netmhcpan_ba', 'Netmhcpan_ba-4.2',
               'Netmhcpan_el', 'Netmhcpan_el-4.2', 'HLAIImaster', 'FusionAttnHLAII']
    
    x = range(len(methods))
    
    # 根据嵌套字典获取每个类别中各方法对应的预测数量
    dp_counts = [data.get('DP', {}).get(m, 0) for m in methods]
    dq_counts = [data.get('DQ', {}).get(m, 0) for m in methods]
    dr_counts = [data.get('DR', {}).get(m, 0) for m in methods]
    
    plt.figure(figsize=(10, 6))
    
    
    plt.plot(x, dp_counts, marker='o', label='DP', color='blue')
    plt.plot(x, dq_counts, marker='o', label='DQ', color='green')
    plt.plot(x, dr_counts, marker='o', label='DR', color='red')
    
    # 设置横坐标标签为方法名，斜着显示以防重叠
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.xlabel("Methods",fontsize=18)
    plt.ylabel("Number",fontsize=18)
    plt.title("HLA DP/DQ/DR Prediction Counts by Method")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype_number.png', bbox_inches="tight")
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype_number.pdf', bbox_inches="tight")
    plt.savefig('/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/figure/supertype/supertype_number.svg', bbox_inches="tight")


# 示例：构造一个性能字典后进行绘图
if __name__ == "__main__":
    
    performance = {
        'DP': {'comblib':68.78 , 'Consensus': 73.81, 'SMM': 75.67, 'Netmhcpan_el': 91.51,
               'Netmhcpan_el-4.2':  90.45, 'Netmhcpan_ba': 81.88, 'Netmhcpan_ba-4.2': 83.79,
               'HLAIImaster': 96.69, 'FusionAttnHLAII': 98.76},
        'DQ': {'comblib': 63.43, 'Consensus': 77.09, 'SMM': 78.40, 'Netmhcpan_el': 68.54,
               'Netmhcpan_el-4.2': 79.02, 'Netmhcpan_ba': 80.63, 'Netmhcpan_ba-4.2': 83.62,
               'HLAIImaster': 85.00, 'FusionAttnHLAII': 90.30},
        'DR': {'comblib': 71.00, 'Consensus': 81.23, 'SMM': 81.63, 'Netmhcpan_el': 83.81,
               'Netmhcpan_el-4.2': 83.30, 'Netmhcpan_ba':88.47, 'Netmhcpan_ba-4.2': 89.22,
               'HLAIImaster': 87.36, 'FusionAttnHLAII': 91.54}
    }
    
    plot_performance(performance)
    
    '''
    # 构造一个示例数据嵌套字典，每个类别下各方法对应的预测数量
    data = {
        'DP': {
            'comblib': 4,
            'Consensus3': 4,
            'SMM': 4,
            'Netmhcpan_ba': 13,
            'Netmhcpan_ba-4.2': 13,
            'Netmhcpan_el': 13,
            'Netmhcpan_el-4.2': 13,
            'HLAIImaster': 13,
            'FusionAttnHLAII': 13
        },
        'DQ': {
            'comblib': 6,
            'Consensus3': 6,
            'SMM': 6,
            'Netmhcpan_ba': 25,
            'Netmhcpan_ba-4.2': 25,
            'Netmhcpan_el': 25,
            'Netmhcpan_el-4.2': 25,
            'HLAIImaster': 25,
            'FusionAttnHLAII': 25
        },
        'DR': {
            'comblib': 5,
            'Consensus3': 26,
            'SMM': 15,
            'Netmhcpan_ba': 46,
            'Netmhcpan_ba-4.2': 46,
            'Netmhcpan_el': 46,
            'Netmhcpan_el-4.2': 46,
            'HLAIImaster': 46,
            'FusionAttnHLAII': 46
        }
    }
    
    plot_hla_prediction_counts(data)
    '''