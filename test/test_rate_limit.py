import re

def parse_rate_limit(limit_string):
    # 定义时间单位到秒的映射
    time_units = {
        's': 1, 'sec': 1, 'second': 1,
        'm': 60, 'min': 60, 'minute': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
        'mo': 2592000, 'month': 2592000,
        'y': 31536000, 'year': 31536000
    }

    # 使用正则表达式匹配数字和单位
    match = re.match(r'^(\d+)/(\w+)$', limit_string)
    if not match:
        raise ValueError(f"Invalid rate limit format: {limit_string}")

    count, unit = match.groups()
    count = int(count)

    # 转换单位到秒
    if unit not in time_units:
        raise ValueError(f"Unknown time unit: {unit}")

    seconds = time_units[unit]

    return (count, seconds)

# 测试函数
test_cases = [
    "2/min", "5/hour", "10/day", "1/second", "3/mo", "1/year",
    "20/s", "15/m", "8/h", "100/d", "50/mo", "2/y"
]

for case in test_cases:
    try:
        result = parse_rate_limit(case)
        print(f"{case} => {result}")
    except ValueError as e:
        print(f"Error parsing {case}: {str(e)}")