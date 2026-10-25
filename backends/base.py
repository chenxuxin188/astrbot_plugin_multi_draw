"""绘图后端的公共部分：请求 / 结果数据结构、错误类型、HTTP 客户端池与通用工具。

本包只依赖标准库与 httpx，不依赖 AstrBot，便于单独测试。
"""

import asyncio
import base64
import binascii
import html
import ipaddress
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

ORIENTATIONS = ("portrait", "landscape", "square")
MAX_IMAGE_BYTES = 64 * 1024 * 1024
SECRET_MASK = "••••"


class DrawError(Exception):
    """绘图失败。message 为可直接展示给用户的中文说明。

    retryable 只在请求确定没有被处理时为 True（连接失败、限流、网关暂时不可用），
    避免超时后重复提交导致重复生成、重复扣费。
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class InputImage:
    """图片修改时用户提供的图片。插件已统一转成 PNG（有透明通道时）或 JPEG，并限制了尺寸。"""

    data: bytes
    mime: str
    width: int = 0
    height: int = 0


@dataclass
class DrawRequest:
    prompt: str
    negative: str = ""
    orientation: str = "portrait"  # portrait / landscape / square
    seed: int = 0  # 0 表示随机（后端不支持种子时忽略）
    # 图片修改：非空时按这些图片和提示词生成修改后的图片
    images: list[InputImage] = field(default_factory=list)
    # 重绘幅度（0-1，NovelAI / ComfyUI 等图生图后端使用），0 表示使用后端配置的默认值
    strength: float = 0.0

    @property
    def is_edit(self) -> bool:
        return bool(self.images)


def image_orientation(image: InputImage) -> str:
    """按图片宽高判断方向，宽高接近时视为方图；不知道尺寸时返回空串。"""
    if not image.width or not image.height:
        return ""
    ratio = image.width / image.height
    return "landscape" if ratio > 1.15 else "portrait" if ratio < 0.87 else "square"


@dataclass
class BalanceInfo:
    """余额查询结果：items 为 (名称, 数值) 列表，如 ("剩余额度", "12.34 美元")。"""

    items: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


@dataclass
class DrawResult:
    image: bytes
    mime: str
    # 实际使用的参数（模型、尺寸、种子等），写入绘图记录
    params: dict = field(default_factory=dict)
    # 后端随图片返回的文字（如 Gemini 的说明、OpenAI 的 revised_prompt），仅记录到日志
    text: str = ""


# ---------------------------------------------------------------- 配置读取


def as_str(value, default: str = "") -> str:
    return default if value is None else str(value).strip()


def as_int(value, default: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        number = default
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return number


def as_float(value, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    if number != number or number in (float("inf"), float("-inf")):
        number = default
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return number


def as_bool(value, default: bool) -> bool:
    """读取开关；"false"、"0" 等字符串视为关闭，空串或无法识别的值使用默认值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return default
    return default if value is None else bool(value)


def normalize_base_url(url: str, default: str) -> str:
    """规范化 API 地址：去掉空白和末尾的 /，未写协议时补 https://，留空时使用默认地址。"""
    url = (url or "").strip().rstrip("/")
    if not url:
        return default.rstrip("/")
    return url if "://" in url else f"https://{url}"


def normalize_proxy_url(url: str) -> str:
    """规范化代理地址：未写协议时按 http 处理，留空返回空串（使用全局代理或直连）。"""
    url = (url or "").strip()
    if url and "://" not in url:
        url = f"http://{url}"
    return url


def mask_url_credentials(url: str) -> str:
    """隐藏地址中的用户名和密码，例如 http://user:pass@host → http://***@host。"""
    return re.sub(r"//[^/@]+@", "//***@", url or "")


def mask_secret(secret: str) -> str:
    """API Key 等敏感值的展示形式：只保留末 4 位。"""
    secret = str(secret or "")
    if not secret:
        return ""
    return SECRET_MASK + (secret[-4:] if len(secret) > 8 else "")


def parse_extra_json(text, label: str) -> dict:
    """解析配置中的「额外请求参数」JSON 对象；留空返回 {}，已经是 dict 时返回其副本。"""
    if isinstance(text, dict):
        return dict(text)
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DrawError(f"{label} 的额外请求参数不是有效的 JSON：{e}") from e
    if not isinstance(data, dict):
        raise DrawError(f"{label} 的额外请求参数必须是 JSON 对象（{{...}}）")
    return data


# 尺寸、宽高比中单个数值的上限：超过视为无法解析，避免超长数字导致计算溢出
MAX_DIMENSION = 100_000


def parse_size(text: str, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    """解析 "832x1216" 形式的尺寸，失败时返回 default。"""
    try:
        w, h = str(text).lower().replace("×", "x").replace("*", "x").split("x", 1)
        w, h = w.strip(), h.strip()
        if len(w) > 6 or len(h) > 6:
            return default
        width, height = int(w), int(h)
    except (ValueError, AttributeError):
        return default
    if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
        return default
    return width, height


def parse_ratio(text: str) -> tuple[int, int] | None:
    """解析 "3:4" 形式的宽高比，失败时返回 None。"""
    try:
        w, h = str(text).replace("：", ":").split(":", 1)
        w, h = w.strip(), h.strip()
        if len(w) > 6 or len(h) > 6:
            return None
        width, height = int(w), int(h)
    except (ValueError, AttributeError):
        return None
    return (width, height) if 0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION else None


# ---------------------------------------------------------------- 图片数据


def detect_mime(data: bytes) -> str:
    """按文件头识别图片格式，无法识别时返回空串。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    return ""


def decode_b64_image(value: str, label: str) -> bytes:
    """解码 base64 图片，兼容 data:image/png;base64, 前缀，并校验确实是图片。"""
    text = str(value or "").strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        data = base64.b64decode(text + "=" * (-len(text) % 4), validate=False)
    except (binascii.Error, ValueError) as e:
        raise DrawError(f"{label} 返回的图片数据无法解码") from e
    if not data:
        raise DrawError(f"{label} 返回了空的图片数据")
    if len(data) > MAX_IMAGE_BYTES:
        raise DrawError(f"{label} 返回的图片超过 64 MiB")
    if not detect_mime(data):
        raise DrawError(f"{label} 返回的数据不是可识别的图片")
    return data


def image_result(data: bytes, params: dict, text: str = "") -> DrawResult:
    return DrawResult(image=data, mime=detect_mime(data) or "image/png", params=params, text=text)


def _count(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def usage_of(images=1, input_tokens=None, output_tokens=None, total_tokens=None,
             cost: float | None = None, currency: str = "") -> dict:
    """把平台响应中的用量整理成统一格式，作为 params["usage"] 写入绘图记录。

    images 为生成的图片数；*_tokens 为平台返回的 token 数（按 token 计费的平台）；
    cost / currency 为平台直接返回的本次费用（目前只有 xAI）。没有返回的项不写入。
    """
    usage: dict = {"images": _count(images) or 1}
    for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                       ("total_tokens", total_tokens)):
        number = _count(value)
        if number is not None:
            usage[key] = number
    if "total_tokens" not in usage and ("input_tokens" in usage or "output_tokens" in usage):
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 and currency:
        usage.update(cost=round(float(cost), 6), currency=currency)
    return usage


# ---------------------------------------------------------------- HTTP


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)

_STATUS_TEXT = {
    400: "请求参数有误",
    401: "API Key 无效或已过期",
    402: "账户余额不足或订阅无效",
    403: "没有权限（Key 无权使用该模型，或网络 / 地区限制）",
    404: "接口地址或模型不存在，请检查 API 地址和模型名",
    408: "请求超时",
    413: "请求内容过大",
    422: "请求参数有误",
    429: "请求过于频繁或额度已用尽",
}


def response_detail(body: Any) -> str:
    """从常见的错误响应体中取出说明文字（OpenAI / Google / 方舟 / ComfyUI 等格式）。"""
    detail = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("code") or err.get("type") or "")
            code = err.get("code") or err.get("status")
            if code and str(code) not in detail:
                detail = f"{detail}（{code}）" if detail else str(code)
        elif err:
            detail = str(err)
        if not detail:
            detail = str(body.get("message") or body.get("detail") or body.get("msg") or "")
        if not detail and body:
            detail = json.dumps(body, ensure_ascii=False)
    elif isinstance(body, list) and body:
        detail = response_detail(body[0])
    elif body:
        detail = str(body)
        # Cloudflare / nginx 等返回的 HTML 错误页只保留标题
        if detail.lstrip().startswith("<") or "<html" in detail[:1000].lower():
            match = _TITLE_RE.search(detail)
            detail = html.unescape(match.group(1)) if match else ""
    return " ".join(detail.split())[:300]


# Retry-After 超过该秒数时不自动重试（等待太久，直接把错误告诉用户）
MAX_RETRY_WAIT = 60.0


def _retry_after(headers: httpx.Headers | None) -> float | None:
    """读取 Retry-After / retry-after-ms（秒）；没有或无法解析时返回 None。"""
    headers = headers or {}
    try:
        if headers.get("retry-after-ms"):
            return max(0.0, float(headers["retry-after-ms"]) / 1000)
        return max(0.0, float(headers.get("retry-after", "")))
    except (TypeError, ValueError, OverflowError):
        return None


def error_from_response(label: str, status: int, body: Any, headers: httpx.Headers | None = None) -> DrawError:
    """根据 HTTP 状态码与响应体构造 DrawError。"""
    detail = response_detail(body)
    text = f"{label} 服务器错误" if status >= 500 else f"{label} {_STATUS_TEXT.get(status, '请求失败')}"
    message = f"{text}（HTTP {status}）"
    if detail:
        message += f"：{detail}"
    # 429 表示请求被拒绝、没有开始生成；502/503 通常是网关在转发前失败。
    # 500/504/524 等可能已在服务端开始生成，不自动重试，避免重复扣费。
    retry_after = _retry_after(headers)
    retryable = status in (429, 502, 503) and (retry_after is None or retry_after <= MAX_RETRY_WAIT)
    return DrawError(message, status=status, retryable=retryable, retry_after=retry_after)


_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # Tailscale 等组网工具使用的地址段


def is_local_url(url: str) -> bool:
    """地址是否指向本机或局域网（如本地 ComfyUI）。这类地址不走 AstrBot 的全局代理。

    没有点的主机名（如 Docker 服务名 comfyui）也视为局域网地址。
    """
    try:
        host = (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith((".local", ".lan", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host and ":" not in host
    return ip.is_private or ip.is_loopback or ip.is_link_local or (ip.version == 4 and ip in _CGNAT)


class HttpPool:
    """按代理设置复用 httpx.AsyncClient；close() 时统一关闭。

    未单独设置代理时 httpx 会读取环境变量中的代理（AstrBot 的全局代理通过环境变量生效）；
    访问本机或局域网地址时不使用环境变量中的代理。
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport
        self._clients: dict[tuple[str, bool], httpx.AsyncClient] = {}
        self._closed = False
        self._lock = asyncio.Lock()

    async def get(self, proxy: str = "", *, trust_env: bool = True) -> httpx.AsyncClient:
        proxy = normalize_proxy_url(proxy)
        key = (proxy, trust_env and not proxy)
        async with self._lock:
            if self._closed:
                raise DrawError("插件已停用或正在重载，请求已取消")
            client = self._clients.get(key)
            if client is None:
                try:
                    client = httpx.AsyncClient(
                        proxy=proxy or None,
                        trust_env=key[1],
                        transport=self._transport,
                        follow_redirects=True,
                        headers={"User-Agent": "AstrBot-MultiDraw/1.0"},
                    )
                except (ValueError, TypeError, ImportError, httpx.InvalidURL) as e:
                    # 例如代理协议不受支持、地址格式错误，或 socks5 代理缺少 socksio
                    raise DrawError(f"代理地址无效：{e}") from e
                self._clients[key] = client
            return client

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.aclose()


ErrorMapper = Callable[[str, int, Any, httpx.Headers], DrawError]


async def send(http: HttpPool, method: str, url: str, *, label: str, proxy: str = "", timeout: float = 120,
               on_error: ErrorMapper | None = None, **kwargs) -> httpx.Response:
    """发送请求并把网络异常、非 2xx 响应统一转换成 DrawError。

    on_error(label, status, body, headers) 可以按平台的错误格式自定义非 2xx 响应的说明和是否重试，
    默认使用 error_from_response。
    """
    client = await http.get(proxy, trust_env=not is_local_url(url))
    try:
        resp = await client.request(method, url, timeout=httpx.Timeout(timeout, connect=15), **kwargs)
    except httpx.InvalidURL as e:
        raise DrawError(f"{label} API 地址格式不正确：{mask_url_credentials(str(e))}") from e
    except (httpx.LocalProtocolError, UnicodeEncodeError):
        # 请求头不合法，通常是 API Key 里混入了换行、中文或全角字符。
        # 异常文字里带有完整的请求头（包括 API Key），不能展示给用户或写入日志。
        raise DrawError(f"{label} 请求头格式不正确，请检查 API Key 是否混入了换行、空格、中文或全角字符") from None
    except httpx.UnsupportedProtocol as e:
        raise DrawError(f"{label} API 地址协议不受支持，请使用 http:// 或 https://（{e}）") from e
    except httpx.ProxyError as e:
        # 代理拒绝或无法建立隧道，请求没有到达服务端，可以安全重试
        raise DrawError(f"代理连接失败，请检查代理地址（{e}）", retryable=True) from e
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        # 请求还没有发出，可以安全重试
        raise DrawError(f"无法连接 {label}（{type(e).__name__}: {e}）", retryable=True) from e
    except httpx.TimeoutException as e:
        raise DrawError(f"请求 {label} 超时（{type(e).__name__}），可以在配置中调大超时时间") from e
    except httpx.HTTPError as e:
        raise DrawError(f"与 {label} 的连接中断（{type(e).__name__}: {e}）") from e
    if not 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise (on_error or error_from_response)(label, resp.status_code, body, resp.headers)
    return resp


def json_body(resp: httpx.Response, label: str) -> dict:
    try:
        data = resp.json()
    except ValueError as e:
        snippet = response_detail(resp.text) or "空响应"
        raise DrawError(f"{label} 返回的不是 JSON：{snippet[:120]}") from e
    if not isinstance(data, dict):
        raise DrawError(f"{label} 返回的数据格式异常")
    return data


async def with_retries(func, *, attempts: int = 2, delay: float = 3.0):
    """调用 func()，只在 DrawError.retryable 时重试，最多共 attempts 次。"""
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except DrawError as e:
            if not e.retryable or attempt >= attempts or (e.retry_after or 0) > MAX_RETRY_WAIT:
                raise
            wait = e.retry_after if e.retry_after is not None else delay
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def download_image(http: HttpPool, url: str, *, label: str, proxy: str = "", timeout: float = 120,
                         headers: dict | None = None) -> bytes:
    """下载后端返回的图片 URL（或解码 data URL），并校验确实是图片。"""
    url = str(url or "").strip()
    if url.startswith("data:"):
        return decode_b64_image(url, label)
    resp = await with_retries(
        lambda: send(http, "GET", url, label=label, proxy=proxy, timeout=timeout, headers=headers or {})
    )
    data = resp.content
    if len(data) > MAX_IMAGE_BYTES:
        raise DrawError(f"{label} 返回的图片超过 64 MiB")
    if not detect_mime(data):
        raise DrawError(f"{label} 返回的图片链接内容不是可识别的图片")
    return data


# ---------------------------------------------------------------- 后端基类


@dataclass
class Clients:
    """插件持有的长连接客户端，传给后端使用。"""

    http: HttpPool
    novelai: Any = None  # NovelAIClient，只有 NovelAI 后端使用


class Backend:
    """绘图后端基类。每次绘图都按当前配置新建实例，因此实例本身不保存连接。

    子类需要设置类属性，并实现 missing() 与 generate()；支持列出模型时实现 list_models()，
    支持图片修改时设置 max_edit_images 并在 generate() 中处理 req.images，支持余额查询时实现 balance()。
    """

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    # LLM 生成提示词的方式：tags（Danbooru 英文标签）或 natural（自然语言描述）
    default_prompt_mode: ClassVar[str] = "natural"
    # 接口原生支持负面提示词；不支持时负面内容会以文字形式附加到提示词末尾
    supports_negative: ClassVar[bool] = False
    supports_models: ClassVar[bool] = False
    # 图片修改最多接受的输入图片数；0 表示不支持图片修改
    # 类属性是该平台的最大能力；实例可以按当前配置的模型改成更小的值，调用方应从实例上读取
    max_edit_images: int = 0
    # 图片修改使用重绘幅度（req.strength，图生图）而不是修改指令
    supports_strength: ClassVar[bool] = False
    supports_balance: ClassVar[bool] = False
    # 同时进行的请求数上限；None 表示使用插件的全局设置
    max_concurrency: ClassVar[int | None] = None
    doc_url: ClassVar[str] = ""
    # 查看余额 / 账单的官方页面，不支持余额查询时提示用户去这里查看
    billing_url: str = ""

    def __init__(self, section: dict | None):
        self.section = section if isinstance(section, dict) else {}

    def get(self, name: str, default: str = "") -> str:
        return as_str(self.section.get(name), default) or default

    @property
    def prompt_mode(self) -> str:
        """配置中可以覆盖提示词方式（目前只有 ComfyUI 提供该选项）。"""
        mode = as_str(self.section.get("prompt_mode")).lower()
        return mode if mode in ("tags", "natural") else self.default_prompt_mode

    @property
    def edit_prompt_mode(self) -> str:
        """修改图片时的提示词方式：配置了 edit_prompt_mode 时使用它，否则与绘图相同。

        例如 ComfyUI 绘图用 SDXL（标签），修改图片用 Qwen-Image-Edit 等工作流（自然语言修改指令）。
        """
        mode = as_str(self.section.get("edit_prompt_mode")).lower()
        return mode if mode in ("tags", "natural") else self.prompt_mode

    @property
    def proxy(self) -> str:
        return normalize_proxy_url(as_str(self.section.get("proxy_url")))

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), 180.0, 10.0, 1800.0)

    @property
    def model(self) -> str:
        return as_str(self.section.get("model"))

    def missing(self) -> str:
        """未配置完整时返回给用户的提示，配置完整时返回空串。"""
        raise NotImplementedError

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        raise NotImplementedError

    async def list_models(self, clients: Clients) -> list[str]:
        raise DrawError(f"{self.label} 不支持获取模型列表")

    def edit_ready_missing(self) -> str:
        """修改图片前的基本配置检查（Key、地址等），默认与 missing() 相同。

        ComfyUI 等文生图和修改图片使用不同配置的后端可以覆盖：文生图配置有误不影响修改图片。
        """
        return self.missing()

    def edit_missing(self) -> str:
        """当前配置不能修改图片时返回原因（如所选模型不支持），可以修改时返回空串。"""
        if not self.max_edit_images:
            return f"{self.label} 暂不支持修改图片"
        return ""

    def balance_missing(self) -> str:
        """余额查询缺少的配置（如需要单独的管理密钥），配置完整时返回空串。"""
        if not self.supports_balance:
            where = f"，请到 {self.billing_url} 查看" if self.billing_url else ""
            return f"{self.label} 没有提供查询余额的接口{where}"
        return self.missing()

    async def balance(self, clients: Clients) -> BalanceInfo:
        raise DrawError(self.balance_missing() or f"{self.label} 不支持查询余额")

    def plan(self, req: DrawRequest) -> dict:
        """生成前预计使用的参数，写入绘图记录：生成失败时记录中也有模型和尺寸，成功后以实际参数为准。

        默认记录模型；后端提供 size(orientation) 时一并记录尺寸。
        """
        params = {"model": self.model}
        size = getattr(self, "size", None)
        if callable(size):
            try:
                value = size(req.orientation)
                params["size"] = "x".join(map(str, value)) if isinstance(value, tuple) else str(value)
            except Exception:
                pass
        return params

    def summary(self) -> str:
        """一行说明当前参数，用于状态查询。"""
        return f"{self.label} / {self.model or '未设置模型'}"


def final_prompt(req: DrawRequest, supports_negative: bool) -> str:
    """不支持负面提示词的接口：把负面内容以文字形式附加到提示词末尾。"""
    negative = (req.negative or "").strip()
    if supports_negative or not negative:
        return req.prompt
    return f"{req.prompt}\n\nAvoid: {negative}"
