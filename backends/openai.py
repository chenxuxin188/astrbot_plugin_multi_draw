"""OpenAI 图像生成后端（POST /v1/images/generations、/v1/images/edits），兼容 OpenAI 格式的中转服务。

GPT Image 模型（gpt-image-* / chatgpt-image-*）只返回 b64_json，不接受 response_format 和 style；
中转服务上的其他模型（dall-e-3、flux 等）可能返回 b64_json 或图片链接，两种都能识别。
接口没有负面提示词和种子参数：负面内容以文字形式附加到提示词末尾，种子忽略。

修改图片使用 multipart/form-data：一张图时文件字段为 image，多张时为 image[]（与官方 SDK 一致，最多 16 张）。
没有重绘幅度参数，修改完全由提示词决定。

余额：OpenAI 官方没有查询剩余额度的接口。官方地址可以用管理密钥（Admin API Key）查询本月消费（Costs API）；
中转服务（one-api / new-api 等）用绘图的 API Key 查询它们兼容的 /dashboard/billing 接口（非官方约定）。
"""

import contextlib
import math
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from .base import (
    MAX_IMAGE_BYTES,
    Backend,
    BalanceInfo,
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
    image_orientation,
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
    usage_of,
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

# 图片修改：GPT Image 模型最多 16 张输入图，每张为小于 50MB 的 PNG / JPEG / WebP；提示词最长 32000 字符
MAX_EDIT_IMAGES = 16
MAX_EDIT_IMAGE_BYTES = 50 * 1024 * 1024
MAX_PROMPT_CHARS = 32000
EDIT_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
# 修改图片的尺寸：match 保持第一张图的比例，orientation 使用该方向的尺寸设置，auto 由模型决定
EDIT_SIZE_MODES = ("match", "orientation", "auto")
DEFAULT_EDIT_SIZE = "match"

# 余额查询：auto 按 API 地址选择，relay 中转服务额度接口，admin 官方管理密钥（本月消费），off 关闭
BALANCE_MODES = ("auto", "relay", "admin", "off")
DEFAULT_BALANCE_MODE = "auto"
BILLING_URL = "https://platform.openai.com/settings/organization/billing"
LIMITS_URL = "https://platform.openai.com/settings/organization/limits"
# Costs API 每页最多 31 个按天的数据桶（一个月一页），以防万一最多翻这么多页
COSTS_MAX_PAGES = 6
# one-api / new-api 的不限额度令牌返回的 hard_limit_usd
RELAY_UNLIMITED = 100_000_000
RELAY_NOTE = "中转服务返回的额度（非 OpenAI 官方接口），以中转站后台为准"
POINTS_UNIT = "额度点数"


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


def is_official_host(url: str) -> bool:
    """是否为 OpenAI 官方地址（含 eu.api.openai.com 等区域地址）。"""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "api.openai.com" or host.endswith(".api.openai.com")


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


def match_aspect_size(width: int, height: int, budget: tuple[int, int]) -> str:
    """按输入图的宽高比、以 budget 的像素数为准计算 gpt-image-2 / 2.5 的尺寸。

    例如 1536x2048 的图、1024x1536 的像素预算 → 1088x1456。
    """
    ratio = width / height
    h = math.sqrt(budget[0] * budget[1] / ratio)
    return snap_free_size(max(16, round(h * ratio)), max(16, round(h)))


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


def edit_error(label: str, status: int, body, headers=None) -> DrawError:
    """/images/edits 的错误：中转服务不支持该接口、图片格式或参数不被接受时给出明确说明，其余同 openai_error。"""
    err = body.get("error") if isinstance(body, dict) else None
    err = err if isinstance(err, dict) else {}
    code, param, message = as_str(err.get("code")).lower(), as_str(err.get("param")), as_str(err.get("message"))
    detail = response_detail(body)
    suffix = f"：{detail}" if detail else ""
    # one-api 对 /images/edits 返回 501 api_not_implemented；其他中转服务可能返回 404 / 405
    if code == "api_not_implemented" or status == 501 or (
            status in (404, 405) and code != "model_not_found" and "model" not in message.lower()):
        return DrawError(f"{label} 的 API 地址不支持修改图片（/images/edits，HTTP {status}）{suffix}。"
                         "请换用支持该接口的中转服务，或在「修改图片使用的平台」中换一个平台", status=status)
    if code == "unsupported_file_mimetype":
        return DrawError(f"{label} 不接受这张图片的格式（只支持 PNG / JPEG / WebP）{suffix}", status=status)
    if status == 400 and (code == "unknown_parameter" or message.lower().startswith("unknown parameter")):
        # 例如 moderation：官方 HTTP 文档列出了该参数，但 SDK 的表单格式中没有
        return DrawError(f"{label} 的图片修改接口不接受参数 {param or '（见下方说明）'}，请在 OpenAI 设置中关闭对应选项"
                         f"（如内容审核强度）或从额外请求参数中去掉后再试（HTTP 400）{suffix}", status=status)
    if status == 400 and param == "model" and "dall-e-2" in message.lower():
        # 发送了 response_format / style 时，OpenAI 会误报为 "Invalid value ... Value must be 'dall-e-2'"
        return DrawError(f"{label} 拒绝了图片修改请求：GPT Image 模型不接受 response_format / style 参数，"
                         f"请检查额外请求参数或中转服务的设置（HTTP 400）{suffix}", status=status)
    if status == 413:
        # 中转服务前面的 nginx 默认只允许 1 MB 的请求体，上传图片时很容易超过
        return DrawError(f"{label} 拒绝了上传的图片：请求内容超过了服务器允许的大小（HTTP 413）{suffix}。"
                         "可以减少图片数量，或换用允许上传更大文件的中转服务", status=status)
    error = openai_error(label, status, body, headers)
    # moderation_stage 为 input 时（moderation_details 在 error 内或响应顶层）：输入包括上传的图片
    text = str(error)
    if text.startswith("提示词被"):
        error.args = ("提示词或输入图片被" + text[len("提示词被"):],)
    return error


def admin_error(label: str, status: int, body, headers=None) -> DrawError:
    """管理密钥（Admin API Key）请求的错误：401 / 403 说明需要哪种密钥。"""
    if status in (401, 403):
        detail = response_detail(body)
        return DrawError(f"{label} 管理密钥（Admin API Key）无效或权限不足（HTTP {status}）"
                         + (f"：{detail}" if detail else "")
                         + "。这里需要组织 Owner 在 Admin keys 页面创建的 sk-admin- 开头的密钥，"
                         "普通 API Key 不能查询消费",
                         status=status)
    return openai_error(label, status, body, headers)


# ---------------------------------------------------------------- 余额工具


def month_start(now: datetime) -> int:
    """本月 1 日 0 点（UTC）的 Unix 时间戳。"""
    now = now.astimezone(timezone.utc)
    return int(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def money(value: float, unit: str) -> str:
    return f"{value:.0f} {unit}" if unit == POINTS_UNIT else f"{value:.2f} {unit}"


# ---------------------------------------------------------------- 后端


class OpenAIBackend(Backend):
    key = "openai"
    label = "OpenAI"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    max_edit_images = MAX_EDIT_IMAGES
    supports_strength = False
    supports_balance = True
    max_concurrency = None
    doc_url = "https://developers.openai.com/api/docs/guides/image-generation"
    billing_url = BILLING_URL

    @staticmethod
    def _clean_key(value) -> str:
        key = as_str(value)
        return key[7:].strip() if key.lower().startswith("bearer ") else key

    @property
    def api_key(self) -> str:
        return self._clean_key(self.section.get("api_key"))

    @property
    def balance_key(self) -> str:
        """可选的管理密钥（Admin API Key），只用于在官方 API 上查询本月消费。"""
        return self._clean_key(self.section.get("balance_key"))

    @property
    def base_url(self) -> str:
        return api_base(self.get("base_url"))

    @property
    def model(self) -> str:
        return as_str(self.section.get("model")) or DEFAULT_MODEL

    @property
    def edit_model(self) -> str:
        """修改图片使用的模型，留空时与绘图模型相同。"""
        return as_str(self.section.get("edit_model")) or self.model

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def size(self, orientation: str) -> str:
        return self._orientation_size(self.model, orientation)

    def _orientation_size(self, model: str, orientation: str) -> str:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        return resolve_size(model, orientation, self.get(f"{orientation}_size", DEFAULT_SIZES[orientation]))

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

    def _model_fields(self, family: str) -> dict:
        """按模型系列决定的可选参数（文生图和修改图片共用）。"""
        fields = {}
        quality = self.get("quality", DEFAULT_QUALITY).lower()
        if family in GPT_FAMILIES:
            # GPT Image 模型不接受 response_format / style；不支持的取值直接省略，由服务端使用默认值
            if quality in ("xhigh", "max") and family != "gpt25":
                quality = "high"  # 只有 gpt-image-2.5 支持 xhigh / max，其他模型最高为 high
            if quality in QUALITIES[family]:
                fields["quality"] = quality
            fmt = self.get("output_format", "png").lower()
            fmt = {"jpg": "jpeg"}.get(fmt, fmt)
            if fmt in ("jpeg", "webp"):
                fields["output_format"] = fmt
            if self.get("moderation").lower() == "low":
                fields["moderation"] = "low"
            if self.get("background").lower() == "opaque":
                fields["background"] = "opaque"
        else:
            if family == "dalle3" and quality in ("high", "xhigh", "max"):
                fields["quality"] = "hd"
            response_format = self.get("response_format", "auto").lower()
            if response_format == "auto":
                # DALL·E 默认返回会过期的链接，改要 b64_json；其他模型规则未知，不发送
                # （中转服务可能把改名的 GPT Image 模型转发给 OpenAI，带上该参数会返回 400）
                response_format = "b64_json" if family in ("dalle3", "dalle2") else ""
            if response_format in ("b64_json", "url"):
                fields["response_format"] = response_format
        return fields

    def build_body(self, req: DrawRequest) -> dict:
        model = self.model
        body = {"model": model, "prompt": final_prompt(req, self.supports_negative), "n": 1}
        size = self.size(req.orientation)
        if size:
            body["size"] = size
        body.update(self._model_fields(model_family(model)))
        # 额外请求参数最后合并，值为 null 表示删除该字段
        for key, value in self.extra_body().items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
        return body

    # ------------------------------------------------------------ 图片修改

    def edit_missing(self) -> str:
        model = self.edit_model
        if model_family(model) in ("dalle2", "dalle3"):  # DALL·E 2 / 3 已于 2026-05-12 下线
            return (f"当前模型 {model} 不支持修改图片，请在 OpenAI 设置的「修改图片使用的模型」中改用 "
                    "gpt-image-2.5-sunburst 或 gpt-image-2.5-flare")
        return ""

    @property
    def edit_size_mode(self) -> str:
        mode = self.get("edit_size", DEFAULT_EDIT_SIZE).lower()
        return mode if mode in EDIT_SIZE_MODES else DEFAULT_EDIT_SIZE

    def edit_size(self, req: DrawRequest) -> str:
        """修改图片时的 size；返回空串表示不发送。

        默认保持第一张图的宽高比：gpt-image-2 / 2.5 支持任意尺寸，以该方向尺寸设置的像素数为准按原图比例计算；
        用户要求换方向时（req.orientation 与原图方向不同）使用该方向的尺寸。其他 GPT Image 模型只有三种标准尺寸，
        中转服务上规则未知的模型不发送（通常保持原图比例）。
        """
        model, mode = self.edit_model, self.edit_size_mode
        family = model_family(model)
        if mode == "auto":
            return "auto" if family in GPT_FAMILIES else ""
        orientation = req.orientation if req.orientation in DEFAULT_SIZES else "portrait"
        preset = self._orientation_size(model, orientation)
        if mode == "orientation":
            return preset
        if family not in GPT_FAMILIES:
            return ""
        first = req.images[0] if req.images else None
        if (family in ("gpt25", "gpt2") and first is not None and first.width > 0 and first.height > 0
                and image_orientation(first) == orientation):
            budget = parse_size(preset, (0, 0))
            if not budget[0]:  # 该方向的尺寸设置为 auto：按默认尺寸的像素数计算
                budget = parse_size(DEFAULT_SIZES[orientation])
            return match_aspect_size(first.width, first.height, budget)
        return preset

    def plan(self, req: DrawRequest) -> dict:
        if not req.is_edit:
            return super().plan(req)
        params = {"mode": "edit", "model": self.edit_model}
        with contextlib.suppress(Exception):
            params["size"] = self.edit_size(req) or "auto"
        return params

    def build_edit_form(self, req: DrawRequest) -> tuple[dict, list, int]:
        """构造 multipart 表单，返回 (文本字段, 文件, 超出上限被忽略的图片数)。"""
        model = self.edit_model
        images = req.images[:MAX_EDIT_IMAGES]
        ignored = len(req.images) - len(images)
        # 与官方 SDK 一致：一张图用 image，多张用 image[]（不用 image[0]：new-api 转发时会打乱顺序）
        name = "image" if len(images) == 1 else "image[]"
        files = []
        for i, img in enumerate(images, 1):
            # 按文件内容确定格式：中转服务按文件扩展名推断类型，扩展名必须与内容一致
            mime = detect_mime(img.data)
            ext = EDIT_MIME_EXT.get(mime)
            if not ext:
                raise DrawError(f"第 {i} 张图片的格式（{mime or as_str(img.mime) or '未知'}）{self.label} 不支持，"
                                "只能使用 PNG / JPEG / WebP")
            if len(img.data) >= MAX_EDIT_IMAGE_BYTES:
                raise DrawError(f"第 {i} 张图片超过 50 MB，{self.label} 不接受")
            files.append((name, (f"image_{i}.{ext}", img.data, mime)))
        prompt = final_prompt(req, self.supports_negative)
        if len(prompt) > MAX_PROMPT_CHARS:
            raise DrawError(f"修改说明超过 {MAX_PROMPT_CHARS} 个字符，{self.label} 不接受")
        form = {"model": model, "prompt": prompt, "n": "1"}
        size = self.edit_size(req)
        if size:
            form["size"] = size
        form.update(self._model_fields(model_family(model)))
        # 额外请求参数最后合并：表单字段只能是文字，对象和数组无法发送，忽略；值为 null 表示删除该字段
        for key, value in self.extra_body().items():
            if value is None:
                form.pop(key, None)
            elif isinstance(value, bool):
                form[key] = "true" if value else "false"
            elif isinstance(value, (str, int, float)):
                form[key] = str(value)
        return form, files, ignored

    # ------------------------------------------------------------ 请求

    async def _read_image(self, clients: Clients, resp, on_error=openai_error) -> tuple[bytes, dict, dict]:
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
            raise on_error(self.label, resp.status_code, body, resp.headers)
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
        """中转服务的报错可能原样带上 API Key / 管理密钥，展示给用户、写入日志前打码。"""
        try:
            yield
        except DrawError as e:
            text = str(e)
            for key in (self.api_key, self.balance_key):
                if len(key) >= 8 and key in text:
                    text = text.replace(key, mask_secret(key))
            if text != str(e):
                e.args = (text,)
            raise

    async def _request(self, clients: Clients, method: str, path: str, *, timeout: float, on_error=openai_error,
                       **kwargs):
        """发送请求，只重试确定没有被处理的失败（连接失败、限流 429、503）。"""
        url = f"{self.base_url}{path}"
        return await with_retries(lambda: send(
            clients.http, method, url, label=self.label, proxy=self.proxy, timeout=timeout,
            on_error=on_error, headers=self.headers, **kwargs,
        ))

    @staticmethod
    def _result_params(params: dict, data: dict, body: dict) -> dict:
        """记录实际使用的参数：官方响应会回显实际使用的尺寸、质量和格式（例如 auto 时）。"""
        for key in ("quality", "output_format"):
            value = as_str(data.get(key)) or as_str(body.get(key))
            if value:
                params[key] = value
        for key in ("moderation", "background", "response_format"):
            if body.get(key):
                params[key] = as_str(body[key])
        usage = data.get("usage")  # 官方文档标注 usage 不一定返回；GPT Image 按输入、输出 token 计费
        usage = usage if isinstance(usage, dict) else {}
        items = data.get("data")
        params["usage"] = usage_of(len(items) if isinstance(items, list) else 1, usage.get("input_tokens"),
                                   usage.get("output_tokens"), usage.get("total_tokens"))
        return params

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        if req.is_edit:
            return await self._edit(clients, req)
        self._require()
        body = self.build_body(req)
        with self._mask_key():
            # 不跟随跳转：301 / 302 会把 POST 变成 GET
            resp = await self._request(clients, "POST", "/images/generations", timeout=self.timeout, json=body,
                                       follow_redirects=False)
            image, data, item = await self._read_image(clients, resp)
        params = {
            "model": as_str(body.get("model")),
            "size": as_str(data.get("size")) or as_str(body.get("size")) or "auto",
        }
        return image_result(image, self._result_params(params, data, body), as_str(item.get("revised_prompt")))

    async def _edit(self, clients: Clients, req: DrawRequest) -> DrawResult:
        self._require()
        reason = self.edit_missing()
        if reason:
            raise DrawError(reason)
        form, files, ignored = self.build_edit_form(req)
        with self._mask_key():
            # multipart 的 Content-Type（含 boundary）由 httpx 生成；不跟随跳转，超时和可能已计费的 5xx 不重试
            resp = await self._request(clients, "POST", "/images/edits", timeout=self.timeout, data=form,
                                       files=files, follow_redirects=False, on_error=edit_error)
            image, data, item = await self._read_image(clients, resp, on_error=edit_error)
        params = {
            "mode": "edit",
            "model": as_str(form.get("model")) or self.edit_model,
            "size": as_str(data.get("size")) or as_str(form.get("size")) or "auto",
            "input_images": len(files),
        }
        if ignored:
            params["ignored_images"] = ignored  # 超过 16 张的部分没有发送
        return image_result(image, self._result_params(params, data, form), as_str(item.get("revised_prompt")))

    async def list_models(self, clients: Clients) -> list[str]:
        """列出模型；优先返回图像模型，筛选后为空（中转服务命名不规范）时返回全部。"""
        self._require()
        with self._mask_key():
            resp = await self._request(clients, "GET", "/models", timeout=min(self.timeout, 60.0))
            items = json_body(resp, self.label).get("data")
        items = items if isinstance(items, list) else []
        ids = sorted({as_str(m.get("id")) for m in items if isinstance(m, dict)} - {""})
        return [i for i in ids if is_image_model(i)] or ids

    # ------------------------------------------------------------ 余额

    @property
    def balance_mode(self) -> str:
        mode = self.get("balance_mode", DEFAULT_BALANCE_MODE).lower()
        return mode if mode in BALANCE_MODES else DEFAULT_BALANCE_MODE

    def is_official(self) -> bool:
        try:
            return is_official_host(self.base_url)
        except DrawError:
            return False

    def balance_source(self) -> str:
        """实际使用的查询方式：admin（官方管理密钥）、relay（中转服务额度接口）或 off。"""
        mode = self.balance_mode
        if mode == "auto":
            return "admin" if self.is_official() else "relay"
        return mode

    @property
    def admin_base_url(self) -> str:
        """管理密钥只发送给 OpenAI 官方地址：API 地址是官方地址时使用它，否则使用默认官方地址（经同一代理）。

        管理密钥权限很大，即使 API 地址误写成 http:// 也始终使用 https。
        """
        if self.is_official():
            parts = urlsplit(self.base_url)
            return f"https://{parts.hostname}{parts.path}"
        return DEFAULT_BASE_URL

    def balance_missing(self) -> str:
        source = self.balance_source()
        if source == "off":
            where = BILLING_URL if self.is_official() else "中转服务的后台"
            return f"已在 OpenAI 设置中关闭余额查询，请到 {where} 查看"
        if source == "admin":
            key = self.balance_key
            if not key:
                if self.balance_mode == "admin":
                    return "余额查询方式为「官方管理密钥」，但还没有在 OpenAI 设置中填写管理密钥（Admin API Key）"
                return (f"OpenAI 官方 API 不能用普通 API Key 查询余额，请到 {BILLING_URL} 查看；"
                        "如需在这里查看本月已花费的金额，可以在 OpenAI 设置中填写管理密钥（Admin API Key）")
            if not (key.isascii() and key.isprintable()):
                return "OpenAI 管理密钥（Admin API Key）中含有换行、全角或不可见字符，请重新复制填写"
            return ""
        reason = self.missing()
        if reason:
            return reason
        if self.is_official():  # 只有手动选择「中转服务额度接口」时才会到这里
            return (f"OpenAI 官方 API 没有中转服务的额度接口，请到 {BILLING_URL} 查看，"
                    "或把余额查询方式改为「官方管理密钥」查询本月消费")
        return ""

    async def balance(self, clients: Clients) -> BalanceInfo:
        reason = self.balance_missing()
        if reason:
            raise DrawError(reason)
        with self._mask_key():
            if self.balance_source() == "admin":
                return await self._admin_spend(clients)
            return await self._relay_balance(clients)

    async def _get(self, clients: Clients, url: str, key: str, params: dict | None = None, on_error=openai_error):
        """余额查询的 GET 请求（可以安全重试连接失败、限流和 503）。"""
        return await with_retries(lambda: send(
            clients.http, "GET", url, label=self.label, proxy=self.proxy, timeout=min(self.timeout, 60.0),
            on_error=on_error, headers={"Authorization": f"Bearer {key}"}, params=params or {},
        ))

    async def _admin_spend(self, clients: Clients) -> BalanceInfo:
        """官方 Costs API：本月（UTC）整个组织已花费的金额，以及消费上限（如有）。"""
        base, key = self.admin_base_url, self.balance_key
        total, currency, page = 0.0, "usd", ""
        start = month_start(datetime.now(timezone.utc))
        for _ in range(COSTS_MAX_PAGES):
            params = {"start_time": start, "bucket_width": "1d", "limit": 31}
            if page:
                params["page"] = page
            body = json_body(await self._get(clients, f"{base}/organization/costs", key, params, admin_error),
                             self.label)
            buckets = body.get("data")
            for bucket in buckets if isinstance(buckets, list) else []:
                results = bucket.get("results") if isinstance(bucket, dict) else None
                for result in results if isinstance(results, list) else []:
                    amount = result.get("amount") if isinstance(result, dict) else None
                    if isinstance(amount, dict):
                        total += as_float(amount.get("value"), 0.0)
                        currency = as_str(amount.get("currency")) or currency
            page = as_str(body.get("next_page"))
            if not body.get("has_more") or not page:
                break
        unit = "美元" if currency.lower() == "usd" else currency.upper()
        items = [("本月已花费（整个组织）", money(total, unit))]
        limit: dict = {}
        with contextlib.suppress(DrawError):  # 消费上限只是附加信息，查询失败时不影响结果
            limit = json_body(await self._get(clients, f"{base}/organization/spend_limit", key,
                                              on_error=admin_error), self.label)
        cents = as_float(limit.get("threshold_amount"), 0.0)  # 单位为美分
        enforcement = limit.get("enforcement")
        status = as_str(enforcement.get("status")).lower() if isinstance(enforcement, dict) else ""
        if cents > 0:
            cap = cents / 100
            if status == "enforcing":
                items += [("每月消费上限（已启用）", money(cap, "美元")),
                          ("距离上限还可花费", money(max(0.0, cap - total), "美元"))]
            else:
                items.append(("每月消费预算（未强制执行）", money(cap, "美元")))
        note = ("OpenAI 没有查询剩余额度的接口，这里是本月已花费的金额（UTC 每月 1 日起，含整个组织所有项目和模型，"
                f"数据可能有延迟），不是余额；剩余额度请到 {BILLING_URL} 查看，消费上限在 {LIMITS_URL} 设置")
        return BalanceInfo(items=items, note=note)

    async def _relay_json(self, clients: Clients, path: str, params: dict | None = None) -> dict | None:
        """请求中转服务的额度接口；接口不存在（404 / 405 / 501，或返回网页）时返回 None。"""
        try:
            resp = await self._get(clients, f"{self.base_url}{path}", self.api_key, params)
        except DrawError as e:
            if e.status in (404, 405, 501):
                return None
            if e.status in (401, 403) and "session key" in str(e).lower():  # 反向代理到了官方接口
                raise DrawError(f"该 API 地址把请求转发到了 OpenAI 官方接口，官方不支持用 API Key 查询余额，"
                                f"请到 {BILLING_URL} 查看；也可以把余额查询方式改为「官方管理密钥」查询本月消费",
                                status=e.status) from e
            raise
        try:
            return json_body(resp, self.label)
        except DrawError:
            return None

    async def _relay_balance(self, clients: Clients) -> BalanceInfo:
        """one-api / new-api 等中转服务兼容的 /dashboard/billing/subscription + usage（非官方约定）。"""
        sub = await self._relay_json(clients, "/dashboard/billing/subscription")
        if sub is None:
            return await self._relay_credit_grants(clients)
        if sub.get("error") or "hard_limit_usd" not in sub:  # 中转服务以 HTTP 200 返回错误
            raise DrawError(f"{self.label} 查询额度失败：{response_detail(sub) or '响应中没有额度信息'}")
        today = datetime.now(timezone.utc).date()
        # one-api / new-api 忽略日期、返回累计用量；旧格式要求带上日期（end_date 不含当天）
        usage = await self._relay_json(clients, "/dashboard/billing/usage", {
            "start_date": today.replace(day=1).isoformat(), "end_date": (today + timedelta(days=1)).isoformat()})
        if usage is None or usage.get("error") or "total_usage" not in usage:
            raise DrawError(f"{self.label} 查询用量失败：{response_detail(usage) or '响应中没有用量信息'}")
        unit = await self._relay_unit(clients)
        total = as_float(sub.get("hard_limit_usd"), 0.0)
        used = as_float(usage.get("total_usage"), 0.0) / 100  # total_usage 以 1/100 为单位
        unlimited = total == RELAY_UNLIMITED if unit == POINTS_UNIT else total >= RELAY_UNLIMITED
        note = RELAY_NOTE
        if unlimited:
            # 只表示这个令牌没有单独的额度上限，账户本身的余额可能有限，接口不返回
            items = [("令牌额度", "不限额度"), ("已用", money(used, unit))]
            note = f"该令牌不限额度，账户的剩余额度请到中转服务的后台查看。{RELAY_NOTE}"
        else:
            items = [("剩余额度", money(max(0.0, total - used), unit)), ("已用", money(used, unit)),
                     ("总额度", money(total, unit))]
        until = as_int(sub.get("access_until"), 0)
        if until > 0:
            with contextlib.suppress(ValueError, OverflowError, OSError):
                items.append(("令牌有效期至", datetime.fromtimestamp(until, timezone.utc).strftime("%Y-%m-%d")))
        return BalanceInfo(items=items, note=note)

    async def _relay_credit_grants(self, clients: Clients) -> BalanceInfo:
        """少数旧中转服务只提供 /dashboard/billing/credit_grants。"""
        body = await self._relay_json(clients, "/dashboard/billing/credit_grants")
        if body is None:
            raise DrawError("该 API 地址没有提供查询余额的接口，请到中转服务的后台查看")
        if body.get("error") or "total_available" not in body:
            raise DrawError(f"{self.label} 查询余额失败：{response_detail(body) or '响应中没有余额信息'}")
        items = [("剩余额度", money(as_float(body.get("total_available"), 0.0), "美元"))]
        if body.get("total_used") is not None:
            items.append(("已用", money(as_float(body.get("total_used"), 0.0), "美元")))
        if body.get("total_granted") is not None:
            items.append(("总额度", money(as_float(body.get("total_granted"), 0.0), "美元")))
        return BalanceInfo(items=items, note=RELAY_NOTE)

    async def _relay_unit(self, clients: Clients) -> str:
        """额度单位：new-api 的公开接口 /api/status 返回 quota_display_type（USD / CNY / TOKENS）；
        one-api 关闭 display_in_currency 时返回原始额度点数。查询失败时按美元显示。"""
        parts = urlsplit(self.base_url)
        try:
            # 公开接口，不带 API Key
            resp = await send(clients.http, "GET", f"{parts.scheme}://{parts.netloc}/api/status", label=self.label,
                              proxy=self.proxy, timeout=15.0)
            data = json_body(resp, self.label).get("data")
        except DrawError:
            return "美元"
        if not isinstance(data, dict):
            return "美元"
        kind = as_str(data.get("quota_display_type")).upper()
        if kind == "CNY":
            return "元"
        if kind == "TOKENS" or (not kind and data.get("display_in_currency") is False):
            return POINTS_UNIT
        return "美元"

    def summary(self) -> str:
        text = f"{self.label} / {self.model} / 竖图 {self.size('portrait') or '自动'}"
        return text if self.edit_model == self.model else f"{text} / 修改图片 {self.edit_model}"
