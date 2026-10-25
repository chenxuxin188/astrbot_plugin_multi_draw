"""NovelAI 后端：官方 API 的精简客户端（生成图片 / 图生图修改图片 / 查询订阅与 Anlas）。

本模块只依赖标准库与 httpx，不依赖 AstrBot，便于单独测试。
例外：修改图片时需要把底图缩放成恰好的请求尺寸，标准库无法解码 JPEG，
因此在 prepare_init_image() 中延迟导入 Pillow（AstrBot 自带），不可用时给出可读的错误。
"""

import asyncio
import base64
import dataclasses
import html
import importlib.util
import io
import math
import random
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from .base import (
    MAX_DIMENSION,
    Backend,
    BalanceInfo,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    InputImage,
    as_bool,
    as_float,
    as_int,
    as_str,
    image_orientation,
    image_result,
    is_local_url,
    with_retries,
)

DEFAULT_BASE_URL = "https://image.novelai.net"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)

# Opus 会员免费出图条件：单张、无底图、总像素 ≤ 1024x1024、步数 ≤ 28（V5 模型除外）
OPUS_FREE_MAX_PIXELS = 1024 * 1024
OPUS_FREE_MAX_STEPS = 28
MAX_SEED = 4294967295 - 7

TIER_NAMES = {0: "Paper", 1: "Tablet", 2: "Scroll", 3: "Opus"}

# 图生图（action "img2img"）：官方网页端默认 strength 0.7、noise 0，滑块范围 0.01-0.99 / 0-0.99
EDIT_DEFAULT_STRENGTH = 0.7
EDIT_MIN_STRENGTH = 0.01
EDIT_MAX_STRENGTH = 0.99
EDIT_DEFAULT_NOISE = 0.0
EDIT_MAX_NOISE = 0.99
# 官方网页端 validateParameters：宽 × 高 ≤ 3145728，步数 ≤ 50；尺寸是 64 的倍数
NAI_MAX_PIXELS = 3145728
SIZE_MULTIPLE = 64
# 底图宽高比限制在 1:4 ~ 4:1（官方只限制总像素，单边上限未公开）；超出时居中裁剪，避免极端长边
EDIT_MAX_ASPECT = 4.0
# 底图与目标尺寸的宽高比（取对数）相差不超过该值时直接拉伸（与网页端一致），超过时先居中裁剪
ASPECT_TOLERANCE = 0.03


class NovelAIError(DrawError):
    """NovelAI 请求失败。message 为可直接展示给用户的中文说明。

    retryable 只在请求确定没有被处理时为 True（连接失败、并发锁定、网关暂时不可用），
    避免超时后重复提交导致重复生成、重复扣费。
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 concurrent: bool = False):
        super().__init__(message, status=status, retryable=retryable)
        self.concurrent = concurrent


@dataclass
class ImageRequest:
    model: str
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    scale: float
    cfg_rescale: float
    sampler: str
    noise_schedule: str
    seed: int = 0
    # 图生图的底图：为 None 时是文生图。发送前必须是恰好 width × height 的 PNG（见 prepare_init_image）
    image: bytes | None = field(default=None, repr=False)
    strength: float = EDIT_DEFAULT_STRENGTH
    noise: float = EDIT_DEFAULT_NOISE
    # 用户提供的图片数；NovelAI 只使用第一张，其余的忽略
    input_images: int = 0


def edit_size(src_width: int, src_height: int, budget_pixels: int) -> tuple[int, int]:
    """图生图的输出尺寸：接近原图宽高比（限制在 1:4 ~ 4:1），宽 × 高 ≤ budget_pixels，宽高都是 64 的倍数。

    NovelAI 没有"跟随原图"的尺寸选项，输出尺寸就是请求里的 width × height。
    """
    m = SIZE_MULTIPLE
    budget = max(m * m, min(int(budget_pixels), NAI_MAX_PIXELS))
    ratio = min(max(src_width / src_height, 1 / EDIT_MAX_ASPECT), EDIT_MAX_ASPECT)
    target = math.log(ratio)
    best, best_key = (m, m), None
    # 逐个尝试 64 倍数的宽度（最多约 60 个），每个宽度取最接近目标比例的几个高度
    # （高度超过约 2100 时 3% 的容差能容纳两个 64 的台阶，所以多试一个 h0 + 2m）
    for w in range(m, int(math.sqrt(budget * EDIT_MAX_ASPECT)) + 2 * m, m):
        h_max = budget // w // m * m
        if h_max < m:
            break
        h0 = int(w / ratio) // m * m
        for h in {max(m, min(h0, h_max)), min(h0 + m, h_max), min(h0 + 2 * m, h_max)}:
            # 宽高比相差 3% 以内的候选取面积最大的；都超出时取宽高比最接近的
            key = (max(0.0, abs(math.log(w / h) - target) - ASPECT_TOLERANCE), -w * h)
            if best_key is None or key < best_key:
                best, best_key = (w, h), key
    return best


def prepare_init_image(data: bytes, width: int, height: int, keep_alpha: bool) -> bytes:
    """按官方网页端（fillImageBackground）的做法整理底图：缩放（Lanczos）到恰好 width × height，
    不支持透明图的模型（V5 以外）把透明背景铺成白色，编码为 PNG。
    宽高比相差超过 3% 时先居中裁剪，避免画面变形。
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise NovelAIError("NovelAI 修改图片需要 Pillow 来调整原图尺寸，请在 AstrBot 环境中安装 Pillow") from e
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im.load()
            has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
            im = im.convert("RGBA" if has_alpha else "RGB")
            src_ratio, dst_ratio = im.width / im.height, width / height
            if abs(math.log(src_ratio / dst_ratio)) > ASPECT_TOLERANCE:
                if src_ratio > dst_ratio:
                    crop_w = max(1, round(im.height * dst_ratio))
                    left = (im.width - crop_w) // 2
                    im = im.crop((left, 0, left + crop_w, im.height))
                else:
                    crop_h = max(1, round(im.width / dst_ratio))
                    top = (im.height - crop_h) // 2
                    im = im.crop((0, top, im.width, top + crop_h))
            if im.size != (width, height):
                lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                im = im.resize((width, height), lanczos)
            if has_alpha and not keep_alpha:
                canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(canvas, im).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "PNG")
            return buf.getvalue()
    except Exception as e:
        raise NovelAIError(f"无法读取要修改的图片（{type(e).__name__}）") from e


def normalize_base_url(url: str) -> str:
    """规范化 API 地址：去掉空白和末尾的 /，未写协议时补 https://，留空时使用官方地址。"""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
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


def parse_size(text: str, default: tuple[int, int] = (832, 1216)) -> tuple[int, int]:
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


def fit_size(width: int, height: int, *, free_guard: bool = True) -> tuple[int, int]:
    """把尺寸对齐到 64 的倍数；开启 free_guard 时按比例缩小到 Opus 免费像素上限内。"""
    width, height = max(64, int(width)), max(64, int(height))
    if free_guard and width * height > OPUS_FREE_MAX_PIXELS:
        ratio = math.sqrt(OPUS_FREE_MAX_PIXELS / (width * height))
        width, height = width * ratio, height * ratio
    return max(64, int(width) // 64 * 64), max(64, int(height) // 64 * 64)


def is_v5_model(model: str) -> bool:
    return str(model).startswith("nai-diffusion-5")


def is_opus_free(model: str, width: int, height: int, steps: int) -> bool:
    """参数是否满足 Opus 免费出图条件（不含订阅状态判断）。"""
    return (
        not is_v5_model(model)
        and width * height <= OPUS_FREE_MAX_PIXELS
        and steps <= OPUS_FREE_MAX_STEPS
    )


def random_seed() -> int:
    return random.randint(1, MAX_SEED)


def build_payload(req: ImageRequest) -> dict:
    """构造 /ai/generate-image 的请求体（V3 / V4 / V4.5 通用格式）。"""
    seed = req.seed if 0 < req.seed <= MAX_SEED else random_seed()
    params = {
        "params_version": 3,
        "width": req.width,
        "height": req.height,
        "scale": float(req.scale),
        "sampler": req.sampler,
        "steps": int(req.steps),
        "n_samples": 1,
        "seed": seed,
        "noise_schedule": req.noise_schedule,
        "cfg_rescale": float(req.cfg_rescale),
        "ucPreset": 0,
        "qualityToggle": False,
        "sm": False,
        "sm_dyn": False,
        "dynamic_thresholding": False,
        "controlnet_strength": 1,
        "legacy": False,
        "legacy_v3_extend": False,
        "legacy_uc": False,
        "add_original_image": True,
        "skip_cfg_above_sigma": None,
        "use_coords": False,
        "normalize_reference_strength_multiple": True,
        "characterPrompts": [],
        "negative_prompt": req.negative,
        "v4_prompt": {
            "caption": {"base_caption": req.prompt, "char_captions": []},
            "use_coords": False,
            "use_order": True,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": req.negative, "char_captions": []},
            "legacy_uc": False,
        },
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": [],
    }
    if req.sampler == "k_euler_ancestral":
        params["deliberate_euler_ancestral_bug"] = False
        params["prefer_brownian"] = True
    action = "generate"
    if req.image is not None:
        # 与官方网页端一致：action 为 img2img，底图是不带 data: 前缀的 base64 PNG，
        # strength / noise 放在 parameters 顶层（嵌套的 parameters.img2img 只用于局部重绘 infill），
        # 图生图关闭 SMEA，extra_noise_seed 默认取 seed - 1，color_correct 为 False。
        action = "img2img"
        params.update({
            "image": base64.b64encode(req.image).decode("ascii"),
            "strength": clamp_strength(req.strength),
            "noise": round(as_float(req.noise, EDIT_DEFAULT_NOISE, 0.0, EDIT_MAX_NOISE), 2),
            "extra_noise_seed": seed - 1 if seed > 1 else seed + 1,
            "color_correct": False,
            "sm": False,
            "sm_dyn": False,
        })
    return {"input": req.prompt, "model": req.model, "action": action, "parameters": params}


def clamp_strength(value: object) -> float:
    """重绘幅度：0 或无效值使用官方默认 0.7，其余限制在 0.01-0.99（官方网页端滑块范围）。"""
    strength = as_float(value, 0.0)
    if strength <= 0:
        strength = EDIT_DEFAULT_STRENGTH
    return round(min(max(strength, EDIT_MIN_STRENGTH), EDIT_MAX_STRENGTH), 2)


def extract_image(body: bytes) -> bytes:
    """从生成接口的响应中取出图片（通常是包含 image_0.png 的 ZIP）。"""
    if body[:8] == b"\x89PNG\r\n\x1a\n" or body[:3] == b"\xff\xd8\xff" or body[:4] == b"RIFF":
        return body
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise NovelAIError("NovelAI 返回了空的压缩包")
            return zf.read(names[0])
    except zipfile.BadZipFile as e:
        raise NovelAIError("NovelAI 返回的数据无法解析为图片") from e


_STATUS_TEXT = {
    400: "请求参数有误",
    401: "Token 无效或已过期",
    402: "订阅无效或 Anlas 不足",
    403: "请求被拒绝（可能是网络或地区限制）",
    404: "接口地址不存在，请检查 base_url",
    409: "账户正在进行其它生成",
    413: "请求内容过大（原图太大）",
    429: "账户正在进行其它生成或请求过于频繁",
}


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def error_from_response(status: int, body: object) -> NovelAIError:
    """根据 HTTP 状态码与响应体构造 NovelAIError。"""
    detail = ""
    if isinstance(body, dict):
        detail = str(body.get("message") or body.get("error") or "")
    elif body:
        detail = str(body)
        # Cloudflare 等返回的 HTML 错误页只保留标题
        if detail.lstrip().startswith("<") or "<html" in detail[:1000].lower():
            match = _TITLE_RE.search(detail)
            detail = html.unescape(match.group(1)) if match else ""
    # 接口仍沿用旧名 training steps 表示 Anlas（官方网页端同样替换后展示）
    detail = re.sub(r"training steps", "Anlas", " ".join(detail.split()), flags=re.I)[:200]
    if status >= 500:
        text = "NovelAI 服务器错误"
    else:
        text = _STATUS_TEXT.get(status, "请求失败")
    concurrent = status == 429 or (status == 409 and "concurrent" in detail.lower())
    # 500/504/524 等可能已在服务端开始生成，不自动重试
    retryable = concurrent or status in (502, 503)
    message = f"{text}（HTTP {status}）"
    if detail:
        message += f"：{detail}"
    return NovelAIError(message, status=status, retryable=retryable, concurrent=concurrent)


def anlas_total(subscription: dict) -> int | None:
    """订阅 Anlas + 购买 Anlas；取不到时返回 None。"""
    steps = subscription.get("trainingStepsLeft") if isinstance(subscription, dict) else None
    if not isinstance(steps, dict):
        return None
    fixed, bought = steps.get("fixedTrainingStepsLeft"), steps.get("purchasedTrainingSteps")
    if isinstance(fixed, int) and isinstance(bought, int):
        return fixed + bought
    return None


def tier_name(tier: object) -> str:
    return TIER_NAMES.get(tier, str(tier)) if isinstance(tier, int) else "未知"


def _format_wait(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    return f"{minutes} 分钟" if minutes < 120 else f"{minutes / 60:.1f} 小时"


def balance_info(data: dict) -> BalanceInfo:
    """把 GET /user/subscription 的结果整理成余额信息。

    trainingStepsLeft 就是 Anlas（接口沿用旧名 training steps）：fixedTrainingStepsLeft 为订阅 Anlas
    （续订时补满，订阅结束清零），purchasedTrainingSteps 为购买的 Anlas（不过期）。
    usage 是 Opus 的 V5 用量条：percent 是剩余百分比，purchasedPercent 是另外购买的额外用量
    （不含在 percent 里），isNegative 为 True 时 V5 改扣 Anlas。expiresAt 是秒级时间戳。
    """
    tier, active = data.get("tier"), bool(data.get("active"))
    status = "有效" if active else "未激活"
    expiry = format_expiry(data.get("expiresAt"))
    if active and expiry:
        status += f"，{expiry} 到期"
    items = [("订阅", f"{tier_name(tier)}（{status}）")]
    steps = data.get("trainingStepsLeft") if isinstance(data.get("trainingStepsLeft"), dict) else {}
    total = anlas_total(data)
    if total is not None:
        items.append(("Anlas", f"{total}（订阅 {steps['fixedTrainingStepsLeft']} + 购买 {steps['purchasedTrainingSteps']}）"))
    else:
        items.append(("Anlas", "未知（接口没有返回 Anlas 余额）"))
    usage = data.get("usage")
    # 官方网页端只对有效的 Opus（tier >= 3）订阅显示用量条
    if (isinstance(tier, int) and tier >= 3 and active and isinstance(usage, dict)
            and isinstance(usage.get("percent"), (int, float)) and not isinstance(usage.get("percent"), bool)):
        if usage.get("isNegative"):
            text = "已用尽（V5 改扣 Anlas，恢复后重新免费）"
        else:
            # 与网页端 usageDisplayPercent 一致：最多显示 100%
            text = f"剩余 {min(100, max(0, int(usage['percent'])))}%"
        extra = usage.get("purchasedPercent")
        if isinstance(extra, (int, float)) and extra > 0:
            text += f"，额外用量 +{int(extra)}%"
        wait = usage.get("timeUntilNextPercent")
        if isinstance(wait, (int, float)) and wait > 0:
            text += f"，约 {_format_wait(wait)}后恢复 1%"
        elif wait == 0 and not usage.get("isNegative") and usage["percent"] >= 100:
            # timeUntilNextPercent 为 0 表示用量超过 100%，暂停恢复
            text += "（超过 100% 时暂停恢复）"
        items.append(("V5 用量条", text))
    return BalanceInfo(items=items)


def format_expiry(value: object) -> str:
    """把 expiresAt（秒或毫秒时间戳）格式化为日期，无法解析时返回空串。"""
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    if value > 1e12:
        value /= 1000
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def mask_token(token: str) -> str:
    token = str(token or "")
    if len(token) <= 10:
        return "*" * len(token)
    return f"{token[:4]}…{token[-4:]}"


class NovelAIClient:
    """复用连接的异步客户端；连接参数变化时自动重建。"""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._key: tuple | None = None
        # 配置变更后旧客户端可能仍有请求在进行，统一在 close() 时关闭
        self._stale: list[httpx.AsyncClient] = []
        self._closed = False
        self._lock = asyncio.Lock()

    async def _http(self, base_url: str, proxy: str) -> httpx.AsyncClient:
        base, proxy = normalize_base_url(base_url), normalize_proxy_url(proxy) or None
        # 未单独设置代理时使用环境变量中的全局代理，但本机和局域网地址（如自建中转）直连
        key = (base, proxy, proxy is None and not is_local_url(base))
        async with self._lock:
            if self._closed:
                raise NovelAIError("插件已停用或正在重载，请求已取消")
            if self._client is None or self._key != key:
                try:
                    client = httpx.AsyncClient(
                        base_url=key[0],
                        proxy=key[1],
                        trust_env=key[2],
                        headers={
                            "User-Agent": USER_AGENT,
                            "Origin": "https://novelai.net",
                            "Referer": "https://novelai.net/",
                        },
                        transport=self._transport,
                    )
                except (ValueError, TypeError, ImportError, httpx.InvalidURL) as e:
                    # 例如代理协议不受支持、地址格式错误，或 socks5 代理缺少 socksio
                    raise NovelAIError(f"API 地址或代理地址无效：{e}") from e
                if self._client is not None:
                    self._stale.append(self._client)
                self._client, self._key = client, key
            return self._client

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            clients = self._stale + ([self._client] if self._client is not None else [])
            self._client, self._key, self._stale = None, None, []
        for client in clients:
            await client.aclose()

    async def _request(self, method: str, path: str, *, token: str, base_url: str, proxy: str,
                       timeout: float, **kwargs) -> httpx.Response:
        client = await self._http(base_url, proxy)
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        try:
            resp = await client.request(
                method, path, headers=headers, timeout=httpx.Timeout(timeout, connect=15), **kwargs
            )
        except (httpx.LocalProtocolError, UnicodeEncodeError):
            # 异常文字里带有完整的请求头（包括 Token），不能展示给用户或写入日志
            raise NovelAIError("请求头格式不正确，请检查 NovelAI Token 是否混入了换行、空格、中文或全角字符") from None
        except httpx.InvalidURL as e:
            raise NovelAIError(f"API 地址格式不正确：{mask_url_credentials(str(e))}") from e
        except httpx.UnsupportedProtocol as e:
            raise NovelAIError(f"API 地址协议不受支持，请使用 http:// 或 https://（{e}）") from e
        except httpx.ProxyError as e:
            # 代理拒绝或无法建立隧道，请求没有到达 NovelAI，可以安全重试
            raise NovelAIError(f"代理连接失败，请检查代理地址（{e}）", retryable=True) from e
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            # 请求还没有发出，可以安全重试
            raise NovelAIError(f"无法连接 NovelAI（{type(e).__name__}: {e}）", retryable=True) from e
        except httpx.TimeoutException as e:
            raise NovelAIError(f"请求 NovelAI 超时（{type(e).__name__}）", retryable=False) from e
        except httpx.HTTPError as e:
            raise NovelAIError(f"与 NovelAI 的连接中断（{type(e).__name__}: {e}）", retryable=False) from e
        if not 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise error_from_response(resp.status_code, body)
        return resp

    async def generate(self, req: ImageRequest, *, token: str, base_url: str = DEFAULT_BASE_URL,
                       proxy: str = "", timeout: float = 120, retry_delay: float = 5) -> bytes:
        """生成一张图片并返回图片字节。

        并发锁定（429 / 409 concurrent）最多重试 2 次；连接失败、502/503 重试 1 次。
        超时等无法确定服务端是否已开始生成的错误不重试。
        """
        payload = build_payload(req)
        lock_retries = other_retries = 0
        while True:
            try:
                resp = await self._request(
                    "POST", "/ai/generate-image", token=token, base_url=base_url, proxy=proxy,
                    timeout=timeout, json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/zip"},
                )
                return await asyncio.to_thread(extract_image, resp.content)
            except NovelAIError as e:
                if not e.retryable:
                    raise
                if e.concurrent:
                    lock_retries += 1
                    if lock_retries > 2:
                        raise
                else:
                    other_retries += 1
                    if other_retries > 1:
                        raise
                await asyncio.sleep(retry_delay)

    async def subscription(self, *, token: str, base_url: str = DEFAULT_BASE_URL, proxy: str = "",
                           timeout: float = 20) -> dict:
        """查询订阅信息（含 Anlas 余额）。"""
        resp = await self._request(
            "GET", "/user/subscription", token=token, base_url=base_url, proxy=proxy,
            timeout=timeout, headers={"Accept": "application/json"},
        )
        try:
            data = resp.json()
        except ValueError as e:
            raise NovelAIError("订阅信息不是有效的 JSON") from e
        if not isinstance(data, dict):
            raise NovelAIError("订阅信息格式异常")
        return data


# ---------------------------------------------------------------- 后端


DEFAULT_SIZES = {"portrait": "832x1216", "landscape": "1216x832", "square": "1024x1024"}


class NovelAIBackend(Backend):
    key = "novelai"
    label = "NovelAI"
    default_prompt_mode = "tags"
    supports_negative = True
    # 同一个 NovelAI 账号同时只能生成一张图
    max_concurrency = 1
    doc_url = "https://docs.novelai.net/"
    # 图生图只接受一张底图（parameters.image 是单个字符串）；除局部重绘专用模型外所有模型都支持
    max_edit_images = 1
    supports_strength = True
    # 余额（订阅、Anlas、V5 用量条）用同一个 Persistent API Token 查询 /user/subscription
    supports_balance = True
    billing_url = "https://novelai.net/image"

    @property
    def token(self) -> str:
        # 去掉误粘贴的 "Bearer " 前缀和空白字符
        token = as_str(self.section.get("token"))
        if token.lower().startswith("bearer "):
            token = token[7:]
        return "".join(token.split())

    @property
    def base_url(self) -> str:
        return normalize_base_url(as_str(self.section.get("base_url")))

    @property
    def model(self) -> str:
        return as_str(self.section.get("model")) or "nai-diffusion-4-5-full"

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), 120.0, 10.0, 1800.0)

    @property
    def free_guard(self) -> bool:
        return as_bool(self.section.get("opus_free_guard"), True)

    @property
    def steps(self) -> int:
        steps = as_int(self.section.get("steps"), 28, 1, 50)
        return min(steps, OPUS_FREE_MAX_STEPS) if self.free_guard else steps

    def size(self, orientation: str) -> tuple[int, int]:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        default = DEFAULT_SIZES[orientation]
        text = as_str(self.section.get(f"{orientation}_size"), default) or default
        width, height = parse_size(text, parse_size(default))
        return fit_size(width, height, free_guard=self.free_guard)

    @property
    def edit_noise(self) -> float:
        return as_float(self.section.get("edit_noise"), EDIT_DEFAULT_NOISE, 0.0, EDIT_MAX_NOISE)

    def missing(self) -> str:
        return "" if self.token else "管理员还没有配置 NovelAI Token"

    def edit_missing(self) -> str:
        reason = super().edit_missing()
        if reason:
            return reason
        # *-inpainting 模型只用于局部重绘（action infill，需要遮罩），不能做整图图生图
        if self.model.endswith("-inpainting"):
            return f"当前模型 {self.model} 是局部重绘专用模型，不支持修改图片，请换成对应的普通模型"
        if importlib.util.find_spec("PIL") is None:
            return "NovelAI 修改图片需要 Pillow 来调整原图尺寸，当前环境没有安装 Pillow"
        return ""

    def edit_target_size(self, req: DrawRequest, image: InputImage) -> tuple[int, int]:
        """图生图的输出尺寸（底图会被缩放成这个尺寸）。

        方向与原图一致时（用户没有要求改方向）保持原图宽高比，像素数不超过该方向配置的尺寸
        （开启免费参数保护时 ≤ 1024×1024，否则 ≤ 官方上限 3145728）；
        用户要求改成其它方向、或不知道原图尺寸时使用该方向配置的尺寸，原图居中裁剪。
        """
        source = image_orientation(image)
        orientation = req.orientation if req.orientation in DEFAULT_SIZES else (source or "portrait")
        width, height = self.size(orientation)
        if source and source == orientation:
            return edit_size(image.width, image.height, width * height)
        if width * height > NAI_MAX_PIXELS:
            return edit_size(width, height, NAI_MAX_PIXELS)
        return width, height

    def build_request(self, req: DrawRequest) -> ImageRequest:
        width, height = self.size(req.orientation)
        edit: dict = {}
        if req.is_edit:
            # NovelAI 只接受一张底图，多余的图片忽略（记录在参数 ignored_images 中）
            source = req.images[0]
            width, height = self.edit_target_size(req, source)
            # 原图在 generate() 中再缩放成 width × height 的 PNG，plan() 不做图片处理
            edit = dict(image=source.data, strength=clamp_strength(req.strength), noise=self.edit_noise,
                        input_images=len(req.images))
        return ImageRequest(
            model=self.model,
            prompt=req.prompt,
            negative=req.negative,
            width=width,
            height=height,
            steps=self.steps,
            scale=as_float(self.section.get("scale"), 5.0),
            cfg_rescale=as_float(self.section.get("cfg_rescale"), 0.0),
            sampler=as_str(self.section.get("sampler")) or "k_euler_ancestral",
            noise_schedule=as_str(self.section.get("noise_schedule")) or "karras",
            seed=req.seed if 0 < req.seed <= MAX_SEED else random_seed(),
            **edit,
        )

    def _image_request(self, req: DrawRequest) -> ImageRequest:
        """同一个请求只构造一次，保证记录中的随机种子与实际发送的一致。"""
        cached = getattr(self, "_planned", None)
        if cached is None or cached[0] is not req:
            cached = self._planned = (req, self.build_request(req))
        return cached[1]

    @staticmethod
    def _params(image_req: ImageRequest) -> dict:
        params = {
            "model": image_req.model,
            "size": f"{image_req.width}x{image_req.height}",
            "steps": image_req.steps,
            "scale": image_req.scale,
            "sampler": image_req.sampler,
            "seed": image_req.seed,
            "opus_free": is_opus_free(image_req.model, image_req.width, image_req.height, image_req.steps),
        }
        if image_req.image is not None:
            # 官方订阅说明：有底图的生成不在 Opus 免费范围内；官方网页端的计价却不排除底图。
            # 两者冲突，这里保守地记为"可能消耗 Anlas"，实际是否扣费以前后余额差（anlas_cost）为准
            params.update(mode="edit", input_images=1, strength=clamp_strength(image_req.strength),
                          noise=round(image_req.noise, 2), opus_free=False)
            if image_req.input_images > 1:
                params["ignored_images"] = image_req.input_images - 1
        return params

    def plan(self, req: DrawRequest) -> dict:
        return self._params(self._image_request(req))

    @staticmethod
    def _client(clients: Clients) -> "NovelAIClient":
        if clients.novelai is None:
            raise NovelAIError("NovelAI 客户端未初始化（插件可能正在重载），请稍后再试")
        return clients.novelai

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        image_req = self._image_request(req)
        send_req = image_req
        if image_req.image is not None:
            reason = self.edit_missing()
            if reason:
                raise NovelAIError(reason)
            # 官方网页端发送前把底图缩放到恰好 width × height；V5 支持透明图，保留透明通道
            png = await asyncio.to_thread(
                prepare_init_image, image_req.image, image_req.width, image_req.height,
                is_v5_model(image_req.model),
            )
            send_req = dataclasses.replace(image_req, image=png)
        try:
            image = await self._client(clients).generate(
                send_req, token=self.token, base_url=self.base_url, proxy=self.proxy, timeout=self.timeout
            )
        except NovelAIError as e:
            if image_req.image is not None and e.status == 402:
                raise NovelAIError(f"{e}。修改图片（图生图）可能不在 Opus 免费出图范围内，需要消耗 Anlas",
                                   status=e.status) from e
            raise
        return image_result(image, self._params(image_req))

    async def subscription(self, clients: Clients) -> dict:
        return await self._client(clients).subscription(token=self.token, base_url=self.base_url, proxy=self.proxy)

    async def balance(self, clients: Clients) -> BalanceInfo:
        reason = self.balance_missing()
        if reason:
            raise NovelAIError(reason)
        # GET 查询可以安全重试（只在连接失败、并发锁定等 retryable 错误时重试一次）
        data = await with_retries(lambda: self.subscription(clients))
        return balance_info(data)

    def summary(self) -> str:
        width, height = self.size("portrait")
        return f"NovelAI / {self.model} / 竖图 {width}x{height} / {self.steps} 步"
