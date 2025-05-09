import string
import secrets
from datetime import datetime, timezone, timedelta

def generate_api_key():
    # Define the character set (only alphanumeric)
    chars = string.ascii_letters + string.digits
    # Generate a random string of 36 characters
    random_string = ''.join(secrets.choice(chars) for _ in range(48))
    api_key = "sk-" + random_string
    return api_key

if __name__ == "__main__":
    num_to_generate = 30  # 您可以在这里修改要生成的密钥配置数量
    indent_prefix = "  "  # 指定每一行输出的前缀缩进，例如 "  " (两个空格)

    all_yaml_entries = []
    for _ in range(num_to_generate):
        api_key = generate_api_key()
        print(api_key)
        # Get current time in UTC+8
        tz_utc_8 = timezone(timedelta(hours=8))
        created_at_time = datetime.now(tz_utc_8).strftime('%Y-%m-%dT%H:%M:%S%z')
        # Manually format the timezone offset to +08:00 if needed
        if created_at_time.endswith('+0800'):
            created_at_time = created_at_time[:-2] + ':' + created_at_time[-2:]

        # 定义单个条目的基础 YAML 结构 (无前缀缩进)
        base_yaml_entry_str = f"""- api: {api_key}
  model:
    - powerhunter/gemini-2.5-pro
    - powerhunter/gemini-2.5-flash
    - gemini-t1/*
  preferences:
    SCHEDULING_ALGORITHM: fixed_priority
    AUTO_RETRY: true
    rate_limit: 10/min
    credits: 1
    created_at: {created_at_time}\n\n"""

        # 为基础 YAML 条目的每一行添加指定的前缀缩进
        indented_lines = [f"{indent_prefix}{line}" for line in base_yaml_entry_str.splitlines()]
        yaml_entry = "\n".join(indented_lines)

        all_yaml_entries.append(yaml_entry)

    final_yaml_output = "\n".join(all_yaml_entries)
    print(final_yaml_output)
