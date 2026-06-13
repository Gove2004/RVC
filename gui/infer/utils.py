"""推理 GUI 工具函数"""


def format_error_message(error: Exception | str) -> str:
    """格式化错误消息，只保留最后一行有意义的内容

    Args:
        error: 异常对象或错误字符串

    Returns:
        str: 格式化后的错误消息
    """
    msg = str(error).strip()
    lines = msg.splitlines()
    return lines[-1] if lines else "未知错误"
