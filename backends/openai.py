"""OpenAI 图像生成后端（POST /v1/images/generations），兼容 OpenAI 格式的中转服务。

GPT Image 模型（gpt-image-* / chatgpt-image-*）只返回 b64_json，不接受 response_format 和 style；
中转服务上的其他模型（dall-e-3、flux 等）可能返回 b64_json 或图片链接，两种都能识别。
接口没有负面提示词和种子参数：负面内容以文字形式附加到提示词末尾，种子忽略。
"""

import contextlib
import math
from urllib.parse import urlsplit

from .base import (
    MAX_IMAGE_BYTES,
    Backend,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    as_float,
    as_int,
    as_str,
    decode_b64_image,
    detect_mime,
    download_image,
    error_from_response,
    final_prompt,
    image_result,
    json_body,
    mask_secret,
    mask_url_credentials,
    normalize_base_url,
    parse_extra_json,
    parse_ratio,
    parse_size,
    response_detail,
    send,
    with_retries,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
# 官方推荐的日常出图模型（最快）；gpt-image-1.5 / 1-mini / chatgpt-image-latest 于 2026-12-01 下线
DEFAULT_MODEL = "gpt-image-2.5-flare"
# 三种标准尺寸，所有 GPT Image 模型都支持
DEFAULT_SIZES = {"portrait": "1024x1536", "landscape": "1536x1024", "square": "1024x1024"}
DEFAULT_QUALITY = "medium"
DEFAULT_TIMEOUT = 300.0

# 用户常把对话接口等完整地址粘贴为 API 地址，去掉这些后缀
_URL_SUFFIXES = ("/chat/completions", "/completions", "/responses", "/images/generations", "/images/edits",
                 "/images", "/models")

# gpt-image-2 / 2.5 的任意尺寸规则：边长为 16 的倍数、长宽比不超过 3:1、边长 ≤ 3840、总像素在范围内
MIN_PIXELS, MAX_PIXELS, MAX_EDGE, MAX_RATIO = 655_360, 8_294_400, 3840, 3
# gpt-image-2 / 2.5 的分辨率档位（2560x1440 以上官方标注为实验性）
TIER_SIZES = {
    "1k": DEFAULT_SIZES,
    "2k": {"portrait": "1152x2048", "landscape": "2048x1152", "square": "2048x2048"},
    "4k": {"portrait": "2160x3840", "landscape": "3840x2160", "square": "2880x2880"},
}
# 中转服务上的 DALL·E 3 只接受这三种尺寸
DALLE3_SIZES = {"portrait": "1024x1792", "landscape": "1792x1024", "square": "1024x1024"}

GPT_FAMILIES = ("gpt25", "gpt2", "gpt")
QUALITIES = {
    "gpt25": ("low", "medium", "high", "xhigh", "max"),
    "gpt2": ("low", "medium", "high"),
    "gpt": ("low", "medium", "high"),
}
MODERATION_CODES = ("moderation_blocked", "content_policy_violation")
# 账单 / 额度类 429 需要人工处理，不能重试；error.code 和 error.type 都可能是这些值
BILLING_CODES = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "billing_hard_limit_reached",
)
# 中转服务的额度类 429 没有统一的 code，按说明文字判断（例如 insufficient_user_quota、用户额度不足）。
# 不能用 billing：OpenAI 普通限流的说明里也带有 account/billing 链接
BILLING_WORDS = ("quota", "额度", "余额")


# ---------------------------------------------------------------- 地址与模型


def api_base(url: str) -> str:
    """规范化 API 地址：只填域名时补 /v1，去掉误粘贴的 /chat/completions、/images/generations 等后缀。"""
    base = normalize_base_url(url, DEFAULT_BASE_URL)
    try:
        parts = urlsplit(base)
        valid = parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
        parts.port  # 端口不是数字或超出范围时抛出 ValueError
    except ValueError:
        valid = False
    if not valid:
        raise DrawError(f"OpenAI API 地址无效，请使用 http:// 或 https:// 开头的地址：{mask_url_credentials(base)}")
    for suffix in _URL_SUFFIXES:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return base if urlsplit(base).path.strip("/") else f"{base}/v1"


def model_family(model: str) -> str:
    """按模型名判断参数规则：gpt25 / gpt2（任意尺寸）、gpt（三种标准尺寸）、dalle3、dalle2、other（其他模型）。"""
    # 中转服务可能使用 openai/gpt-image-2 这样带前缀的名字
    name = as_str(model).lower().rsplit("/", 1)[-1]
    if name.startswith("gpt-image-2.5"):
        return "gpt25"
    if name.startswith("gpt-image-2"):
        return "gpt2"
    if name.startswith(("gpt-image", "chatgpt-image")):
        return "gpt"
    if name.startswith("dall-e-3"):
        return "dalle3"
    if name.startswith("dall-e-2"):
        return "dalle2"
    return "other"


def is_image_model(model_id: str) -> bool:
    name = model_id.lower()
    return "image" in name or name.rsplit("/", 1)[-1].startswith("dall-e")


# ---------------------------------------------------------------- 尺寸


def _valid_free_size(width: int, height: int) -> bool:
    return (width % 16 == 0 and height % 16 == 0 and max(width, height) <= MAX_EDGE
            and max(width, height) <= MAX_RATIO * min(width, height) and MIN_PIXELS <= width * height <= MAX_PIXELS)


def _nearest(width: int, height: int, presets: dict) -> str:
    ratio = width / height
    return presets["landscape" if ratio > 1.15 else "portrait" if ratio < 0.87 else "square"]


def snap_free_size(width: int, height: int) -> str:
    """把任意宽高调整为 gpt-image-2 / 2.5 接受的最接近尺寸（尽量保持比例和像素数）。"""
    if _valid_free_size(width, height):
        return f"{width}x{height}"
    ratio = min(max(width / height, 1 / MAX_RATIO), MAX_RATIO)
    pixels = min(max(width * height, MIN_PIXELS), MAX_PIXELS)
    h = math.sqrt(pixels / ratio)
    scale = min(1.0, MAX_EDGE / max(h * ratio, h))
    w0, h0 = round(h * ratio * scale / 16), round(h * scale / 16)
    candidates = [((w0 + dw) * 16, (h0 + dh) * 16) for dw in range(-6, 7) for dh in range(-6, 7)]
    valid = [(w, h) for w, h in candidates if w > 0 and h > 0 and _valid_free_size(w, h)]
    if not valid:  # 理论上不会发生
        return _nearest(width, height, DEFAULT_SIZES)
    w, h = min(valid, key=lambda s: abs(math.log(s[0] / s[1] / ratio)) * 4 + abs(math.log(s[0] * s[1] / pixels)))
    return f"{w}x{h}"


def _ratio_size(rw: int, rh: int) -> tuple[int, int]:
    """把 3:4 这样的比例换算成宽高：方图 1024x1024，其他长边 1536（与默认尺寸一致）。"""
    if rw == rh:
        return 1024, 1024
    long, short = max(rw, rh), min(rw, rh)
    edge = max(16, (1536 * short + 8 * long) // (16 * long) * 16)
    return (1536, edge) if rw > rh else (edge, 1536)


def resolve_size(model: str, orientation: str, text: str) -> str:
    """把配置的尺寸转换成该模型接受的 size；返回空串表示不发送 size（由服务端决定）。

    支持 1024x1536、3:4、1K / 2K / 4K 和 auto；无法识别时使用该方向的默认尺寸
    （中转服务上的其他模型规则未知，无法识别的值原样发送）。
    """
    family = model_family(model)
    orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
    raw = as_str(text)
    key = raw.lower().replace(" ", "")
    if key == "auto":
        return "auto" if family in GPT_FAMILIES else ""
    if key in TIER_SIZES and family != "other":
        key = (TIER_SIZES[key] if family in ("gpt25", "gpt2") else DEFAULT_SIZES)[orientation]
    width, height = parse_size(key, (0, 0))
    if not width and (ratio := parse_ratio(key)):
        width, height = _ratio_size(*ratio)
    if not 0 < max(width, height) <= 100_000:
        if family == "other" and raw:
            return raw  # 中转服务上的其他模型规则未知，原样发送（例如 2K）
        width, height = parse_size(DEFAULT_SIZES[orientation])
    if family in ("gpt25", "gpt2"):
        return snap_free_size(width, height)
    if family == "gpt":
        return _nearest(width, height, DEFAULT_SIZES)
    if family == "dalle3":
        return _nearest(width, height, DALLE3_SIZES)
    if family == "dalle2":
        return f"{width}x{height}" if width == height and width in (256, 512, 1024) else "1024x1024"
    return f"{width}x{height}"


# ---------------------------------------------------------------- 错误


def openai_error(label: str, status: int, body, headers=None) -> DrawError:
    """OpenAI 的错误响应：安全拦截给出友好提示；账单 / 额度类 429 不重试，只重试限流 429 和 503。

    其他 5xx（包括中转服务的 502 / 504）可能已经开始生成并计费，不自动重试。
    """
    err = body.get("error") if isinstance(body, dict) else None
    err = err if isinstance(err, dict) else {}
    code, kind = as_str(err.get("code")).lower(), as_str(err.get("type")).lower()
    if code in MODERATION_CODES or (kind == "image_generation_user_error" and "moderation" in code):
        details = err.get("moderation_details") or body.get("moderation_details")
        details = details if isinstance(details, dict) else {}
        stage = as_str(details.get("moderation_stage")).lower()
        if stage == "output":
            message = f"生成的图片没有通过 {label} 的安全审核"
        elif stage == "input":
            message = f"提示词被 {label} 的安全系统拦截了"
        else:
            message = f"这次绘图被 {label} 的安全系统拦截了"
        categories = details.get("categories")
        if isinstance(categories, list) and categories:
            message += f"（类别：{'、'.join(as_str(c) for c in categories[:5])}）"
        return DrawError(f"{message}，请换一种描述再试", status=status)
    if status < 300:  # 中转服务可能以 HTTP 200 返回错误
        return DrawError(f"{label} 返回了错误：{response_detail(body) or '未知错误'}", status=status)
    if status < 400:  # 生成请求不跟随跳转（301 / 302 会把 POST 变成 GET，得到令人困惑的 404 / 405）
        location = mask_url_credentials(as_str((headers or {}).get("location")))
        return DrawError(f"{label} 的 API 地址发生了跳转（HTTP {status}" + (f" → {location}" if location else "")
                         + "），请检查 API 地址是否正确（例如应使用 https://）", status=status)
    error = error_from_response(label, status, body, headers)
    detail = response_detail(body)
    text = f"{code} {kind} {detail}".lower()
    if status == 429 and (code in BILLING_CODES or kind in BILLING_CODES or any(w in text for w in BILLING_WORDS)):
        return DrawError(f"{label} 账户余额不足或已达到用量 / 消费上限，请到后台检查账单（HTTP 429）"
                         + (f"：{detail}" if detail else ""), status=status)
    if status == 403 and "verif" in str(error).lower():
        return DrawError(f"{error}。GPT Image 模型需要先在 OpenAI 后台完成组织验证（Organization Verification）",
                         status=status)
    error.retryable = status in (429, 503)
    return error


# ---------------------------------------------------------------- 后端


class OpenAIBackend(Backend):
    key = "openai"
    label = "OpenAI"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    max_concurrency = None
    doc_url = "https://developers.openai.com/api/docs/guides/image-generation"

    @property
    def api_key(self) -> str:
        key = as_str(self.section.get("api_key"))
        return key[7:].strip() if key.lower().startswith("bearer ") else key

    @property
    def base_url(self) -> str:
        return api_base(self.get("base_url"))

    @property
    def model(self) -> str:
        return as_str(self.section.get("model")) or DEFAULT_MODEL

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def size(self, orientation: str) -> str:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        return resolve_size(self.model, orientation, self.get(f"{orientation}_size", DEFAULT_SIZES[orientation]))

    def missing(self) -> str:
        if not self.api_key:
            return "管理员还没有配置 OpenAI API Key"
        # 含换行等字符时 httpx 的报错会带上完整的请求头（即 API Key），含全角字符时无法发送
        if not (self.api_key.isascii() and self.api_key.isprintable()):
            return "OpenAI API Key 中含有换行、全角或不可见字符，请重新复制填写"
        try:
            self.base_url
        except DrawError as e:
            return str(e)
        return ""

    def _require(self) -> None:
        reason = self.missing()
        if reason:
            raise DrawError(reason)

    def extra_body(self) -> dict:
        value = self.section.get("extra_body")
        return dict(value) if isinstance(value, dict) else parse_extra_json(as_str(value), self.label)

    def build_body(self, req: DrawRequest) -> dict:
        model = self.model
        family = model_family(model)
        body = {"model": model, "prompt": final_prompt(req, self.supports_negative), "n": 1}
        size = self.size(req.orientation)
        if size:
            body["size"] = size
        quality = self.get("quality", DEFAULT_QUALITY).lower()
        if family in GPT_FAMILIES:
            # GPT Image 模型不接受 response_format / style；不支持的取值直接省略，由服务端使用默认值
            if quality in ("xhigh", "max") and family != "gpt25":
                quality = "high"  # 只有 gpt-image-2.5 支持 xhigh / max，其他模型最高为 high
            if quality in QUALITIES[family]:
                body["quality"] = quality
            fmt = self.get("output_format", "png").lower()
            fmt = {"jpg": "jpeg"}.get(fmt, fmt)
            if fmt in ("jpeg", "webp"):
                body["output_format"] = fmt
            if self.get("moderation").lower() == "low":
                body["moderation"] = "low"
            if self.get("background").lower() == "opaque":
                body["background"] = "opaque"
        else:
            if family == "dalle3" and quality in ("high", "xhigh", "max"):
                body["quality"] = "hd"
            response_format = self.get("response_format", "auto").lower()
            if response_format == "auto":
                # DALL·E 默认返回会过期的链接，改要 b64_json；其他模型规则未知，不发送
                # （中转服务可能把改名的 GPT Image 模型转发给 OpenAI，带上该参数会返回 400）
                response_format = "b64_json" if family in ("dalle3", "dalle2") else ""
            if response_format in ("b64_json", "url"):
                body["response_format"] = response_format
        # 额外请求参数最后合并，值为 null 表示删除该字段
        for key, value in self.extra_body().items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
        return body

    async def _read_image(self, clients: Clients, resp) -> tuple[bytes, dict, dict]:
        """从响应中取出图片，返回 (图片, 响应 JSON, 图片所在的 data 项)。"""
        if resp.headers.get("content-type", "").lower().startswith("image/"):
            data = resp.content  # 少数中转服务直接返回图片
            if len(data) > MAX_IMAGE_BYTES or not detect_mime(data):
                raise DrawError(f"{self.label} 返回的数据不是可识别的图片")
            return data, {}, {}
        try:
            body = json_body(resp, self.label)
        except DrawError as e:
            raise DrawError(f"{e}（请检查 API 地址是否正确）") from e
        items = body.get("data")
        items = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        if body.get("error") and not items:
            raise openai_error(self.label, resp.status_code, body, resp.headers)
        for item in items:
            b64 = item.get("b64_json") or item.get("b64Json")
            if isinstance(b64, str) and b64.strip():
                return decode_b64_image(b64, self.label), body, item
            url = as_str(item.get("url"))
            if url:
                image = await download_image(clients.http, url, label=self.label, proxy=self.proxy,
                                             timeout=self.timeout)
                return image, body, item
        failed = next((item for item in items if item.get("error")), None)
        detail = response_detail(failed or body)
        raise DrawError(f"{self.label} 的响应中没有图片" + (f"：{detail[:200]}" if detail else ""))

    @contextlib.contextmanager
    def _mask_key(self):
        """中转服务的报错可能原样带上 API Key，展示给用户、写入日志前打码。"""
        try:
            yield
        except DrawError as e:
            key = self.api_key
            if len(key) >= 8 and key in str(e):
                e.args = (str(e).replace(key, mask_secret(key)),)
            raise

    async def _request(self, clients: Clients, method: str, path: str, *, timeout: float, **kwargs):
        """发送请求，只重试确定没有被处理的失败（连接失败、限流 429、503）。"""
        url = f"{self.base_url}{path}"
        return await with_retries(lambda: send(
            clients.http, method, url, label=self.label, proxy=self.proxy, timeout=timeout,
            on_error=openai_error, headers=self.headers, **kwargs,
        ))

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        self._require()
        body = self.build_body(req)
        with self._mask_key():
            # 不跟随跳转：301 / 302 会把 POST 变成 GET
            resp = await self._request(clients, "POST", "/images/generations", timeout=self.timeout, json=body,
                                       follow_redirects=False)
            image, data, item = await self._read_image(clients, resp)
        # 官方响应会回显实际使用的尺寸和质量（例如 auto 时）
        params = {
            "model": as_str(body.get("model")),
            "size": as_str(data.get("size")) or as_str(body.get("size")) or "auto",
        }
        for key in ("quality", "output_format"):
            value = as_str(data.get(key)) or as_str(body.get(key))
            if value:
                params[key] = value
        for key in ("moderation", "background", "response_format"):
            if body.get(key):
                params[key] = as_str(body[key])
        usage = data.get("usage")
        if isinstance(usage, dict) and usage.get("total_tokens") is not None:
            params["tokens"] = as_int(usage.get("total_tokens"), 0)
        return image_result(image, params, as_str(item.get("revised_prompt")))

    async def list_models(self, clients: Clients) -> list[str]:
        """列出模型；优先返回图像模型，筛选后为空（中转服务命名不规范）时返回全部。"""
        self._require()
        with self._mask_key():
            resp = await self._request(clients, "GET", "/models", timeout=min(self.timeout, 60.0))
            items = json_body(resp, self.label).get("data")
        items = items if isinstance(items, list) else []
        ids = sorted({as_str(m.get("id")) for m in items if isinstance(m, dict)} - {""})
        return [i for i in ids if is_image_model(i)] or ids

    def summary(self) -> str:
        return f"{self.label} / {self.model} / 竖图 {self.size('portrait') or '自动'}"
