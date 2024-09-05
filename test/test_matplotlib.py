import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

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

if __name__ == '__main__':
    create_pic(request_arrivals, 'POST /v1/chat/completions')