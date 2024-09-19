import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

import matplotlib.font_manager as fm
font_path = '/System/Library/Fonts/PingFang.ttc'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

with open('./test/states.json') as f:
    data = json.load(f)
    request_arrivals = data["request_arrivals"]

def create_pic(request_arrivals, key):
    request_arrivals = request_arrivals[key]
    # 将字符串转换为datetime对象
    datetimes = [datetime.fromisoformat(t) for t in request_arrivals]
    # 获取最新的时间
    latest_time = max(datetimes)

    # 创建24小时的时间范围
    time_range = [latest_time - timedelta(hours=i) for i in range(32, 0, -1)]
    # 统计每小时的请求数
    hourly_counts = defaultdict(int)
    for dt in datetimes:
        for t in time_range[::-1]:
            if dt >= t:
                hourly_counts[t] += 1
                break

    # 准备绘图数据
    hours = [t.strftime('%Y-%m-%d %H:00') for t in time_range]
    counts = [hourly_counts[t] for t in time_range]

    # 创建柱状图
    plt.figure(figsize=(15, 6))
    plt.bar(hours, counts)
    plt.title(f'{key} 端点请求量 (过去24小时)')
    plt.xlabel('时间')
    plt.ylabel('请求数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{key.replace("/", "")}.png')

def create_pie_chart(model_counts):
    models = list(model_counts.keys())
    counts = list(model_counts.values())

    # 设置颜色和排列顺序
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    sorted_data = sorted(zip(counts, models, colors), reverse=True)
    counts, models, colors = zip(*sorted_data)

    # 创建饼图
    fig, ax = plt.subplots(figsize=(16, 10))
    wedges, _ = ax.pie(counts, colors=colors, startangle=90, wedgeprops=dict(width=0.5))

    # 添加圆环效果
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    fig.gca().add_artist(centre_circle)

    # 计算总数
    total = sum(counts)

    # 准备标注
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0)

    left_labels = []
    right_labels = []

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))

        percentage = counts[i] / total * 100
        label = f"{models[i]}: {percentage:.1f}%"

        if x > 0:
            right_labels.append((x, y, label))
        else:
            left_labels.append((x, y, label))

    # 绘制左侧标注
    for i, (x, y, label) in enumerate(left_labels):
        ax.annotate(label, xy=(x, y), xytext=(-1.2, 0.9 - i * 0.15), **kw)

    # 绘制右侧标注
    for i, (x, y, label) in enumerate(right_labels):
        ax.annotate(label, xy=(x, y), xytext=(1.2, 0.9 - i * 0.15), **kw)

    plt.title("各模型使用次数对比", size=16)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('model_usage_pie_chart.png', bbox_inches='tight', pad_inches=0.5)

if __name__ == '__main__':
    model_counts = {
        "model_counts": {
            "claude-3-5-sonnet": 94,
            "o1-preview": 71,
            "gpt-4o": 512,
            "gpt-4o-mini": 5,
            "gemini-1.5-pro": 5,
            "deepseek-chat": 7,
            "grok-2-mini": 1,
            "grok-2": 9,
            "o1-mini": 8
        }
    }
    # create_pic(request_arrivals, 'POST /v1/chat/completions')

    create_pie_chart(model_counts["model_counts"])
