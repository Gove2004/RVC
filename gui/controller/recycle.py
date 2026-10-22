"""把文件移入系统回收站（Windows Shell API 封装）。

为什么独立成模块：这是纯 OS FFI，与「离线转换流程」毫无关系。放在
OfflineConversion 里会让一个 controller 同时承担转换状态机与 Win32 结构体
定义两种职责；抽出来之后本模块可被任何"删除产物"场景复用，也便于单独测试。

为什么不用 os.remove：那是永久删除、不可恢复。回收站删除给误删留退路——
离线转换的输出是用户可能正在听/正在比的音频，一次误点不该造成数据损失。
"""
import os


def recycle_file(path: str) -> None:
    """把文件移入回收站（SHFileOperationW + FOF_ALLOWUNDO）。

    非 Windows 平台抛 NotImplementedError（本应用仅 Windows）。

    Raises:
        OSError: Shell API 返回非零或操作被中止。
    """
    if os.name != "nt":
        raise NotImplementedError("回收站删除仅支持 Windows")

    import ctypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", ctypes.c_int),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400

    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    # c_wchar_p 会在末尾补 \0，字符串自带一个 \0 ⇒ 恰为双零结尾（列表语义）
    op.pFrom = path + "\0"
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    ret = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if ret != 0 or op.fAnyOperationsAborted:
        raise OSError(f"SHFileOperationW 返回 {ret}")
