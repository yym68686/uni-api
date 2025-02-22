import os

# 环境变量配置
DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
DEBUG = bool(os.getenv("DEBUG", False))
THINK_PHASE_ACTIVE = os.getenv("THINK_PHASE_ACTIVE", "true").lower() == "true"
DISABLE_DATABASE = os.getenv("DISABLE_DATABASE", "false").lower() == "true" 
