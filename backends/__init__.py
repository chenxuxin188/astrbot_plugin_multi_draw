"""绘图后端注册表。"""

from .base import Backend, Clients, DrawError, DrawRequest, DrawResult, HttpPool
from .comfyui import ComfyUIBackend
from .gemini import GeminiBackend
from .novelai import NovelAIBackend
from .openai import OpenAIBackend
from .seedream import SeedreamBackend
from .sensenova import SenseNovaBackend
from .xai import XAIBackend

BACKEND_ORDER = ("novelai", "openai", "seedream", "xai", "gemini", "sensenova", "comfyui")
BACKENDS: dict[str, type[Backend]] = {
    cls.key: cls
    for cls in (NovelAIBackend, OpenAIBackend, SeedreamBackend, XAIBackend, GeminiBackend, SenseNovaBackend,
                ComfyUIBackend)
}

# /draw use 指令接受的别名
ALIASES = {
    "nai": "novelai", "novel": "novelai",
    "gpt": "openai", "gpt-image": "openai", "dalle": "openai",
    "doubao": "seedream", "豆包": "seedream", "火山": "seedream", "方舟": "seedream", "ark": "seedream",
    "grok": "xai", "x": "xai",
    "google": "gemini", "nano": "gemini", "nanobanana": "gemini", "banana": "gemini", "谷歌": "gemini",
    "商汤": "sensenova", "日日新": "sensenova", "sense": "sensenova",
    "comfy": "comfyui",
}


def resolve_key(text: str) -> str:
    """把用户输入的平台名转换成后端 key，无法识别时返回空串。"""
    text = str(text or "").strip().lower().replace(" ", "")
    if text in BACKENDS:
        return text
    return ALIASES.get(text, "")


def create_backend(key: str, section: dict | None) -> Backend:
    cls = BACKENDS.get(key)
    if cls is None:
        raise DrawError(f"未知的绘图平台：{key}")
    return cls(section)


def backend_missing(backend: Backend) -> str:
    """missing() 出错时返回错误说明，避免一个平台的错误配置影响状态查询和设置页面。"""
    try:
        return backend.missing()
    except Exception as e:
        return f"{backend.label} 配置有误：{str(e).strip() or type(e).__name__}"


def backend_summary(backend: Backend) -> str:
    try:
        return backend.summary()
    except Exception as e:
        return f"{backend.label}（配置有误：{str(e).strip() or type(e).__name__}）"


__all__ = [
    "ALIASES",
    "BACKENDS",
    "BACKEND_ORDER",
    "Backend",
    "Clients",
    "DrawError",
    "DrawRequest",
    "DrawResult",
    "HttpPool",
    "backend_missing",
    "backend_summary",
    "create_backend",
    "resolve_key",
]
