import logging
import os
import sys

def setup_logging():
    """
    配置全局日志记录器。
    此函数应在应用启动时（例如 main.py）被调用一次。
    """
    # 避免重复配置（例如被多次 import 或在测试中重复调用）
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    # 获取日志级别（优先环境变量；如存在 settings 则以 settings 为准）
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        from .config import settings  # 延迟导入，避免因配置缺失导致日志不可用

        log_level = getattr(settings, "log_level", log_level).upper()
    except Exception:
        # 配置加载失败时，仍然允许日志系统工作
        pass
    
    # 获取与 "log_level" 字符串匹配的日志级别对象
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {settings.log_level}")

    # 定义一个新的、更好的日志格式
    # %(name)s 会显示 logger 的名字 (例如 "src.backend.services.agent_service")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # stream=sys.stdout 确保日志输出到标准输出，这在容器部署中更友好
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        stream=sys.stdout 
    )
    
    # 打印一条日志，确认配置已生效
    logging.getLogger(__name__).info(f"日志系统已初始化，级别: {log_level}")
