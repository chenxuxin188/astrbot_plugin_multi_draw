"""Gemini（Nano Banana）文生图 / 图片修改后端。

默认调用 generateContent（POST {base}/models/{model}:generateContent，请求头 x-goog-api-key）；
模型不在 generateContent 上提供时（HTTP 404，不计费）自动改用 Interactions API 再试一次。
尺寸只能通过固定的宽高比（imageConfig.aspectRatio，如 "2:3"）和分辨率档位（imageSize：512 / 1K / 2K / 4K）
控制，按所选模型支持的范围自动修正。文档：https://ai.google.dev/gemini-api/docs/image-generation

修改图片没有单独的接口：同一个请求里先放修改指令文字，再按用户顺序放输入图片（generateContent 用 inlineData，
Interactions 用 {"type": "image", "mime_type", "data"}），base64 不带 data: 前缀。没有重绘幅度、遮罩等字段。
Gemini API 没有查询余额的接口，预付费余额只能在 AI Studio 结算页查看。
"""

import base64
import math
import re
from typing import NamedTuple
from urllib.parse import urlparse

from .base import (
    MAX_DIMENSION,
    Backend,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    InputImage,
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
    normalize_base_url,
    parse_extra_json,
    parse_ratio,
    parse_size,
    response_detail,
    send,
    usage_of,
    with_retries,
)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-nano-banana-2.1"
DEFAULT_TIMEOUT = 300.0
# 2:3 最接近 NovelAI 的 832x1216，1K 档输出 848x1264
DEFAULT_SIZES = {"portrait": "2:3", "landscape": "3:2", "square": "1:1"}
API_CHOICES = ("auto", "generate_content", "interactions")

RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
RATIOS_EXT = RATIOS + ("1:4", "4:1", "1:8", "8:1")
TIERS = ("512", "1K", "2K", "4K")


class Caps(NamedTuple):
    ratios: tuple[str, ...]
    sizes: tuple[str, ...]  # 为空时不发送 imageSize
    thinking: tuple[str, ...]  # 为空时不发送思考等级
    edit_images: int = 14  # 修改图片时最多使用的输入图片数，0 表示不能修改图片


# 各模型支持的比例、分辨率档位、思考等级（官方图片生成指南，2026-10-09）与修改图片的输入图片数：
# 指南「Mix up to 14 reference images」；gemini-2.5-flash-image 只写「works best with up to 3 images」，按 3 张使用
MODEL_CAPS = {
    "gemini-nano-banana-2.1": Caps(RATIOS_EXT, ("1K", "2K", "4K"), ("minimal", "medium", "high")),
    "gemini-3.1-flash-image": Caps(RATIOS_EXT, TIERS, ("minimal", "high")),
    "gemini-3.1-flash-lite-image": Caps(RATIOS, ("1K",), ("minimal", "high")),
    "gemini-3-pro-image": Caps(RATIOS, ("1K", "2K", "4K"), ()),
    "nano-banana-pro-preview": Caps(RATIOS, ("1K", "2K", "4K"), ()),
    "gemini-2.5-flash-image": Caps(RATIOS, (), (), 3),
}
# 未知模型：只用基础比例，不发送思考等级；修改图片按目前所有 Nano Banana 模型的 14 张
DEFAULT_CAPS = Caps(RATIOS, ("1K", "2K", "4K"), ())
MAX_EDIT_IMAGES = max(caps.edit_images for caps in (*MODEL_CAPS.values(), DEFAULT_CAPS))

# 输入图片格式：三份官方列表（Blob 示例、Interactions ImageContent 枚举、图片理解指南）都有的格式
INPUT_MIMES = ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif")
# detect_mime 认不出文件头的格式：只有这些才使用声明的 MIME（声明为 PNG / JPEG / WebP 但文件头不符时数据有误）
UNSNIFFED_MIMES = ("image/heic", "image/heif")
# 内联数据上限「100 MB per request or payload」（file-input-methods；changelog 2026-01-08 从 20MB 提高到 100MB）。
# 不确定按 10^6 还是 2^20 计算，请求体里还有 JSON 和提示词，base64 合计按 95,000,000 字符截止
MAX_INLINE_B64 = 95_000_000

# Gemini API 没有余额 / 用量接口（discovery 文档里没有 billing、quota、balance 资源），只能在 AI Studio 查看
BILLING_URL = "https://aistudio.google.com/billing"  # 预付费余额、交易记录、购买额度、自动充值
USAGE_URL = "https://aistudio.google.com/usage"  # Dashboard > Usage
SPEND_URL = "https://aistudio.google.com/spend"  # 项目每月支出上限

# 已停用的模型 → 官方建议的替代模型（docs/deprecations，2026-10-22）；Imagen 已全部停用
REPLACEMENTS = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image",
    "gemini-3.1-flash-image-preview": "gemini-nano-banana-2.1",
    "gemini-2.5-flash-image-preview": "gemini-nano-banana-2.1",
    "gemini-2.0-flash-preview-image-generation": "gemini-nano-banana-2.1",
}

# generateContent 的 blockReason / finishReason 与 Interactions 的 error.code 中表示被策略拦截的值（小写）
BLOCK_REASONS = {
    "safety", "recitation", "language", "prohibited_content", "spii", "blocklist", "jailbreak",
    "image_safety", "image_prohibited_content", "image_recitation", "image_other", "content_blocked",
    "escalation", "pup_limited_disabled",
}


# ---------------------------------------------------------------- 参数换算


def model_caps(model: str) -> Caps:
    model = model.lower()
    if model in MODEL_CAPS:
        return MODEL_CAPS[model]
    for name in sorted(MODEL_CAPS, key=len, reverse=True):  # 如以后出现的 -001 等版本后缀
        if model.startswith(name):
            return MODEL_CAPS[name]
    return DEFAULT_CAPS


def nearest_ratio(width: float, height: float, allowed: tuple[str, ...]) -> str:
    """在模型支持的比例中找最接近的一个（按对数距离，不做约分，避免出现 683:384 这类无效比例）。"""
    target = math.log(width) - math.log(height)  # 不先相除：超大整数相除会溢出

    def distance(ratio: str) -> float:
        w, h = ratio.split(":")
        return abs(math.log(int(w)) - math.log(int(h)) - target)

    return min(allowed, key=distance)


def normalize_tier(value) -> str:
    """把用户填写的分辨率档位规范成 512 / 1K / 2K / 4K（K 必须大写），无法识别时返回空串。"""
    text = as_str(value).upper().replace(" ", "").replace("PX", "")
    text = {"0.5K": "512", ".5K": "512", "1": "1K", "2": "2K", "4": "4K",
            "1024": "1K", "2048": "2K", "4096": "4K"}.get(text, text)
    return text if text in TIERS else ""


def pick_tier(wanted: str, supported: tuple[str, ...]) -> str:
    """模型不支持该档位时，取不高于它的最大档位；都没有时取最低档。"""
    if not wanted or not supported:
        return ""
    if wanted in supported:
        return wanted
    lower = [s for s in supported if TIERS.index(s) <= TIERS.index(wanted)]
    return lower[-1] if lower else supported[0]


def tier_for_area(area: int, supported: tuple[str, ...]) -> str:
    """按像素面积推断档位：各档位在任意比例下约为 0.26 / 1.05 / 4.2 / 16.8 MP。"""
    if area <= 520_000:
        wanted = "512" if "512" in supported else "1K"
    elif area <= 2_100_000:
        wanted = "1K"
    elif area <= 8_400_000:
        wanted = "2K"
    else:
        wanted = "4K"
    return pick_tier(wanted, supported)


def encode_inputs(images: list[InputImage], label: str = "Gemini") -> list[tuple[str, str]]:
    """把输入图片编码成 [(MIME, 标准 base64)]，保持用户给出的顺序。

    MIME 按文件头识别（不用声明的 image/jpg、application/octet-stream 等接口不认的值）；
    base64 不带 data: 前缀（带前缀时 generateContent 返回 400 Base64 decoding failed）。
    格式不支持或合计过大时在发请求前报错。
    """
    encoded, total = [], 0
    for index, image in enumerate(images, 1):
        raw = image.data or b""
        sniffed = detect_mime(raw)
        mime = sniffed or as_str(image.mime).lower()
        if not raw or (not sniffed and mime in INPUT_MIMES and mime not in UNSNIFFED_MIMES):
            # 空数据或声明的格式与文件头不符：发出去只会得到 400，在发请求前报错
            raise DrawError(f"{label} 第 {index} 张图片的数据无法识别，请换一张图片")
        if mime not in INPUT_MIMES:
            raise DrawError(f"{label} 不支持第 {index} 张图片的格式（{mime or '未知'}），请换成 PNG / JPEG / WebP 图片")
        data = base64.b64encode(raw).decode("ascii")
        total += len(data)
        encoded.append((mime, data))
    if total > MAX_INLINE_B64:
        raise DrawError(f"{label} 输入图片合计过大（编码后约 {total / 1_000_000:.0f} MB，上限约 100 MB），"
                        f"请减少图片数量或换用更小的图片")
    return encoded


def size_text(ratio: str, tier: str) -> str:
    """记录和状态中显示的尺寸，如 "2:3 2K"；修改图片且不发送宽高比时显示「跟随原图」。"""
    shown = ratio or "跟随原图"
    return f"{shown} {tier}" if tier else shown


def gemini_base_url(url: str) -> str:
    """规范化 API 地址：去掉误粘贴的查询参数（如 ?key=）和接口路径（/models/...、/interactions、/openai），没有版本号时补上 /v1beta。"""
    base = normalize_base_url(re.sub(r"[?#].*", "", as_str(url), flags=re.S), DEFAULT_BASE_URL)
    base = re.sub(r"/(models(/.*)?|interactions|openai(/.*)?)$", "", base, flags=re.I).rstrip("/")
    if not re.search(r"/v\d+[a-z0-9]*$", base, flags=re.I):
        base += "/v1beta"
    return base


def merge_extra(body: dict, extra: dict) -> dict:
    """把额外请求参数合并进请求体：对象逐层合并，其他值直接覆盖，null 表示删除该字段。"""
    for key, value in extra.items():
        if value is None:
            body.pop(key, None)
        elif isinstance(value, dict):  # 合并进新对象，其中的 null 同样表示删除，也不会改动配置本身
            target = body.get(key)
            body[key] = merge_extra(target if isinstance(target, dict) else {}, value)
        else:
            body[key] = value
    return body


def _list(value) -> list:
    return value if isinstance(value, list) else []


# ---------------------------------------------------------------- 错误与响应


def _refusal(reason: str, detail: str = "") -> DrawError | None:
    """被安全策略拦截或模型没能出图时的说明；其他情况返回 None。都不自动重试。"""
    tail = f"：{detail[:200]}" if detail else ""
    if reason.lower() in BLOCK_REASONS:
        return DrawError(f"Gemini 拒绝生成：内容被安全策略拦截（{reason}），请修改描述后再试{tail}")
    if reason.lower() == "no_image":
        return DrawError(f"Gemini 没能生成图片（{reason}），可以换个描述再试{tail}")
    return None


def _no_image(reason: str, detail: str, status: str = "") -> DrawError:
    refusal = _refusal(reason, detail)
    if refusal:
        return refusal
    tail = f"：{detail[:200]}" if detail else ""
    if reason.lower() == "max_tokens":
        return DrawError(f"Gemini 输出超出长度限制，没有生成图片（MAX_TOKENS）{tail}")
    if status and status != "completed":
        return DrawError(f"Gemini 生成失败（{status}{'，' + reason if reason else ''}）{tail}")
    if detail and reason.upper() in ("", "STOP"):
        return DrawError(f"Gemini 没有生成图片，模型回复：{detail[:200]}")
    return DrawError(f"Gemini 没有返回图片（{reason or '响应中没有图片数据'}）{tail}")


def _retry_delay(value) -> float | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(value or ""))  # google.rpc.RetryInfo，如 "23s"
    return float(match.group(1)) if match else None


_PREPAY_RE = re.compile(r"prepay(ment)? credit|payment_required", re.I)
_OUTSTANDING_RE = re.compile(r"outstanding balance|past[- ]due", re.I)
# Google 前端的经典文字是「Request payload size exceeds the limit: N bytes.」；输入 token 超限也会提到 exceeds the maximum
_PAYLOAD_RE = re.compile(r"payload size|request (entity )?(is )?too large|exceeds the (maximum|limit)", re.I)


def gemini_error(label: str, status: int, body, headers) -> DrawError:
    """把错误响应转换成 DrawError，兼容 generateContent（Google RPC 格式）与 Interactions（字符串 code，
    网关错误外面还包一层数组）两种格式。"""
    text_all = str(body)
    detail = response_detail(body)
    detail_tail = f"：{detail}" if detail else ""
    # 预付费余额用完：文档为 HTTP 402 payment_required，「Don't retry」（docs/billing、api-errors、troubleshooting）；
    # 论坛（非官方）还见过 HTTP 429 RESOURCE_EXHAUSTED「Your prepayment credits are depleted」。都不重试。
    # 普通限流和按支出限流的 429 不含 prepay credit 字样，仍按限流处理。
    if _PREPAY_RE.search(text_all):
        return DrawError(f"{label} 预付费余额不足（已用完或尚未购买预付费额度），请到 {BILLING_URL} 充值或开启自动充值"
                         f"（HTTP {status}）", status=status)
    if status == 402:  # 也可能来自中转服务自己的余额
        return DrawError(f"{label} 账户余额不足（HTTP 402）{detail_tail}。使用官方 API 时请到 {BILLING_URL} 充值，"
                         f"使用中转服务时请到中转服务充值", status=status)
    # Postpay 账号欠费：403 PERMISSION_DENIED 并提到 outstanding balance（docs/troubleshooting）
    if status == 403 and _OUTSTANDING_RE.search(text_all):
        return DrawError(f"{label} 结算账号有未付款项，请到 {BILLING_URL} 或 Google Cloud 结算控制台处理（HTTP 403）"
                         f"{detail_tail}", status=status)
    # 请求过大（多半是输入图片太多或太大）：确切的报错文字未经验证，保留原始说明
    if status == 413 or (status == 400 and _PAYLOAD_RE.search(text_all)):
        return DrawError(f"{label} 请求内容过大（HTTP {status}），修改图片时请减少图片数量或换用更小的图片{detail_tail}",
                         status=status)
    payload = body[0] if isinstance(body, list) and body else body
    err = payload.get("error") if isinstance(payload, dict) else None
    err = err if isinstance(err, dict) else {}
    code = err.get("code")
    code = code if isinstance(code, str) else as_str(err.get("status"))
    message = " ".join(as_str(err.get("message")).split())[:300]
    reasons, delay, per_day = [], None, False
    for detail in _list(err.get("details")):
        if not isinstance(detail, dict):
            continue
        reasons.append(as_str(detail.get("reason")))
        delay = _retry_delay(detail.get("retryDelay")) if delay is None else delay
        for violation in _list(detail.get("violations")):
            if isinstance(violation, dict) and "perday" in as_str(violation.get("quotaId")).lower():
                per_day = True
    text = f"{code} {message} {' '.join(reasons)}".lower()
    tail = f"：{message}" if message else ""

    refusal = _refusal(code, message)
    if refusal:
        refusal.status = status
        return refusal
    if "api_key_invalid" in text or "api key not valid" in text or "api key expired" in text:
        # 无效 Key 返回的是 HTTP 400 而不是 401
        return DrawError(f"{label} API Key 无效或已过期，请检查配置（HTTP {status}）", status=status)
    if "location is not supported" in text or "not available in your country" in text:
        return DrawError(f"{label} 不支持当前服务器所在的地区，请在配置中设置代理（HTTP {status}）{tail}", status=status)
    result = error_from_response(label, status, body, headers)
    if status == 429:
        # 每日配额用完、当前账号没有该模型的额度（limit: 0）时重试没有意义
        if code.lower() == "quota_exceeded" or per_day or "per day" in text or "per_day" in text or "limit: 0" in text:
            return DrawError(f"{label} 额度已用尽（每日配额用完，或当前账号没有该模型的额度，免费层可能需要开通结算）"
                             f"（HTTP 429）{tail}", status=status)
        delay = result.retry_after if delay is None else delay
        if delay is not None and delay > 60:
            return DrawError(f"{label} 请求过于频繁，请约 {math.ceil(delay)} 秒后再试（HTTP 429）{tail}", status=status)
        result.retry_after = delay
    return result


def _not_served(e: DrawError) -> bool:
    """generateContent 不提供该模型（404 或 "not supported for generateContent"），请求没有被处理。"""
    text = str(e).lower()
    return e.status == 404 or (e.status == 400 and "generatecontent" in text and "not supported" in text)


def parse_generate_content(data: dict, edit: bool = False) -> tuple[str, str]:
    """取出 generateContent 响应中最后一张正式图片的 base64（跳过思考过程中的草图）和附带的文字。

    edit 为 True 时（修改图片）输入里还有用户的图片，被拦截的也可能是图片本身。
    """
    feedback = data.get("promptFeedback")
    block = as_str(feedback.get("blockReason")) if isinstance(feedback, dict) else ""
    if block:
        if edit:
            raise DrawError(f"Gemini 拒绝修改：图片或修改要求被安全策略拦截（{block}），请换一张图片或修改要求后再试")
        raise DrawError(f"Gemini 拒绝生成：提示词被安全策略拦截（{block}），请修改描述后再试")
    candidates = [c for c in _list(data.get("candidates")) if isinstance(c, dict)]
    images, texts = [], []
    for cand in candidates:
        content = cand.get("content")
        for part in _list(content.get("parts")) if isinstance(content, dict) else []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            blob = part.get("inlineData") or part.get("inline_data")
            if isinstance(blob, dict) and blob.get("data"):
                images.append(str(blob["data"]))
            elif as_str(part.get("text")):
                texts.append(as_str(part.get("text")))
    text = "\n".join(texts)
    if images:
        return images[-1], text
    first = candidates[0] if candidates else {}
    raise _no_image(as_str(first.get("finishReason")), as_str(first.get("finishMessage")) or text)


def parse_interaction(data: dict) -> tuple[dict, str]:
    """取出 Interactions 响应中最后一个 model_output 图片块（思考步骤的草图在 summary 里，不会取到）。"""
    blocks = []
    for step in _list(data.get("steps")):
        if isinstance(step, dict) and step.get("type") == "model_output":
            blocks += _list(step.get("content"))
    blocks += _list(data.get("outputs"))  # 2026-06 之前的旧响应格式，只有过时的转发服务还会返回
    blocks = [b for b in blocks if isinstance(b, dict)]
    images = [b for b in blocks if b.get("type") == "image" and (b.get("data") or b.get("uri"))]
    text = "\n".join(as_str(b.get("text")) for b in blocks if b.get("type") == "text" and as_str(b.get("text")))
    if images:
        return images[-1], text
    errors = [e for e in _list(data.get("errors")) if isinstance(e, dict)]
    code = next((as_str(e.get("code")) for e in errors if as_str(e.get("code"))), "")
    detail = "；".join(as_str(e.get("message")) for e in errors if as_str(e.get("message"))) or text
    raise _no_image(code, detail, as_str(data.get("status")))


def gemini_usage(data: dict) -> dict:
    """用量：generateContent 的 usageMetadata，或 Interactions 的 usage。

    思考 token 按输出价格计费，计入输出 token（candidatesTokenCount 不含思考 token）。
    """
    meta = data.get("usageMetadata")
    if isinstance(meta, dict):
        fields = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")
    else:
        meta = data.get("usage")
        fields = ("total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens")
    if not isinstance(meta, dict):
        return usage_of()
    prompt, output, thoughts, total = (meta.get(f) for f in fields)
    if isinstance(output, int) and isinstance(thoughts, int):
        output += thoughts
    return usage_of(1, prompt, output, total)


def decode_image(value: str) -> bytes:
    value = str(value)
    if "-" in value or "_" in value:  # URL 安全的 base64
        value = value.translate(str.maketrans("-_", "+/"))
    return decode_b64_image(value, "Gemini")


# ---------------------------------------------------------------- 后端


class GeminiBackend(Backend):
    key = "gemini"
    label = "Gemini"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    max_concurrency = None
    doc_url = "https://ai.google.dev/gemini-api/docs/image-generation"
    # 类上为各模型中最多的输入图片数；实例上按当前模型给出（见 __init__），generate() 也只使用前几张
    max_edit_images = MAX_EDIT_IMAGES
    # 没有重绘幅度 / 遮罩字段（imageConfig.strength 等返回 400 Unknown name），修改靠文字指令
    supports_strength = False
    supports_balance = False
    billing_url = BILLING_URL

    def __init__(self, section: dict | None):
        super().__init__(section)
        # 插件收集图片、设置页面限制试画图片数都按实例上的 max_edit_images：按当前模型给出，
        # 用户就不会发出 5 张图后有 2 张被静默丢弃（gemini-2.5-flash-image 为 3 张）；类上仍是各模型中的最大值
        self.max_edit_images = self.edit_limit

    @property
    def api_key(self) -> str:
        return as_str(self.section.get("api_key"))

    @property
    def base_url(self) -> str:
        return gemini_base_url(as_str(self.section.get("base_url")))

    @property
    def headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    @property
    def configured_model(self) -> str:
        model = as_str(self.section.get("model"))
        return as_str(model[7:] if model.lower().startswith("models/") else model) or DEFAULT_MODEL

    @property
    def model(self) -> str:
        """实际使用的模型：已停用的模型换成官方替代模型。"""
        model = self.configured_model
        if model.lower().startswith("imagen-"):
            return DEFAULT_MODEL
        return REPLACEMENTS.get(model.lower(), model)

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    @property
    def api(self) -> str:
        value = as_str(self.section.get("api")).lower().replace("-", "_")
        value = "generate_content" if value == "generatecontent" else value
        return value if value in API_CHOICES else "auto"

    @property
    def thinking_level(self) -> str:
        """只在模型文档列出该等级时发送（其他模型发送会报错）。"""
        level = as_str(self.section.get("thinking_level")).lower()
        return level if level in model_caps(self.model).thinking else ""

    @property
    def edit_limit(self) -> int:
        """当前模型修改图片时最多使用的输入图片数。"""
        return model_caps(self.model).edit_images

    def edit_missing(self) -> str:
        if not self.edit_limit:
            return f"当前模型 {self.model} 不支持修改图片，请换用 {DEFAULT_MODEL} 等 Nano Banana 模型"
        return ""

    def balance_missing(self) -> str:
        return (f"Gemini API 没有查询余额的接口：预付费余额和交易记录只能在 AI Studio 结算页 {BILLING_URL} 查看，"
                f"用量见 {USAGE_URL}，项目支出上限见 {SPEND_URL}")

    def edit_ratio(self, req: DrawRequest, configured_ratio: str) -> str:
        """修改图片时的宽高比。方向与第一张图一致时取该模型支持的、最接近第一张图的比例（保持原图形状）；
        用户要求换方向时用配置的比例；不知道第一张图的尺寸时返回空串（不发送宽高比，由接口按参考图决定）。"""
        first = req.images[0]
        try:
            width, height = int(first.width or 0), int(first.height or 0)
        except (TypeError, ValueError, OverflowError):
            return ""
        # 尺寸无效（负数、超大）时同样当作不知道尺寸，避免 math.log / 除法抛出非 DrawError 的异常
        if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
            return ""
        shape = image_orientation(InputImage(b"", "", width, height))
        if req.orientation not in DEFAULT_SIZES or shape == req.orientation:
            return nearest_ratio(width, height, model_caps(self.model).ratios)
        return configured_ratio

    def plan(self, req: DrawRequest) -> dict:
        ratio, tier = self.image_config(req.orientation)
        if not req.is_edit:
            return {"model": self.model, "size": size_text(ratio, tier)}
        return {"model": self.model, "size": size_text(self.edit_ratio(req, ratio), tier), "mode": "edit",
                "input_images": min(len(req.images), self.edit_limit)}

    def image_config(self, orientation: str) -> tuple[str, str]:
        """返回 (宽高比, 分辨率档位)；档位为空表示不发送，使用模型默认的 1K。"""
        caps = model_caps(self.model)
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        text = as_str(self.section.get(f"{orientation}_size")).replace("/", ":")
        wanted = normalize_tier(self.section.get("image_size"))
        # 尺寸后面可以带档位（如 "16:9 2K"，与状态中显示的格式相同），优先于「分辨率档位」；只写 "2K" 时用默认比例
        match = re.fullmatch(r"(.*?)\s*\b(512|[124]K)", text, flags=re.I)
        if match and (not match.group(1) or parse_ratio(match.group(1)) or parse_size(match.group(1), (0, 0))[0]):
            text, wanted = match.group(1), match.group(2).upper()
        ratio, inferred = parse_ratio(text), ""
        if ratio is None:
            width, height = parse_size(text, (0, 0))
            if width and height:  # 填的是像素尺寸：换算成最接近的比例，并按面积推断档位
                ratio = (width, height)
                if min(width, height) >= 64:
                    inferred = tier_for_area(width * height, caps.sizes)
        ratio = ratio or parse_ratio(DEFAULT_SIZES[orientation])
        tier = pick_tier(wanted, caps.sizes)
        if not tier and inferred != "1K":  # 1K 是默认档位，不必发送
            tier = inferred
        return nearest_ratio(ratio[0], ratio[1], caps.ratios), tier

    def extra_body(self) -> dict:
        value = self.section.get("extra_body")
        return dict(value) if isinstance(value, dict) else parse_extra_json(as_str(value), self.label)

    def missing(self) -> str:
        key = self.api_key
        if not key:
            return "管理员还没有配置 Gemini API Key"
        # 含换行等字符时 httpx 的报错会带上完整的请求头（即 API Key），含全角字符时无法发送
        if not (key.isascii() and key.isprintable()):
            return "Gemini API Key 中含有换行、全角或不可见字符，请重新复制填写"
        return ""

    def build_request(self, api: str, prompt: str, ratio: str, tier: str, seed: int | None,
                      images: list[tuple[str, str]] | tuple = ()) -> tuple[str, dict]:
        """images 为 encode_inputs() 的结果（修改图片）；ratio 为空串时不发送宽高比。文字在前，图片按用户顺序在后。"""
        thinking = self.thinking_level
        if api == "interactions":
            # 不发送 mime_type（只接受 image/jpeg，发 image/png 会 400）和 delivery（会被拒绝）
            response_format = {"type": "image"}
            if ratio:
                response_format["aspect_ratio"] = ratio
            if tier:
                response_format["image_size"] = tier
            content = prompt
            if images:
                content = [{"type": "text", "text": prompt}] + [
                    {"type": "image", "mime_type": mime, "data": data} for mime, data in images]
            # store=False：否则请求（包括用户的图片）会在 Google 保存 55 天（付费层）/ 1 天（免费层）
            body = {"model": self.model, "input": content, "response_format": response_format, "store": False}
            config = {}
            if thinking:
                config["thinking_level"] = thinking  # Interactions 用小写
            if seed is not None:
                config["seed"] = seed
            if config:
                body["generation_config"] = config
            return f"{self.base_url}/interactions", body
        image_config = {}
        if ratio:
            image_config["aspectRatio"] = ratio
        if tier:
            image_config["imageSize"] = tier
        config = {"responseModalities": ["TEXT", "IMAGE"]}
        if image_config:
            config["imageConfig"] = image_config
        if thinking:
            config["thinkingConfig"] = {"thinkingLevel": thinking.upper()}  # generateContent 用大写
        if seed is not None:
            config["seed"] = seed
        # 单个 user 回合不需要 thoughtSignature；再次修改上一张结果时也作为新的 user 图片发送，不伪造 model 回合
        parts = [{"text": prompt}] + [{"inlineData": {"mimeType": mime, "data": data}} for mime, data in images]
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        return f"{self.base_url}/models/{self.model}:generateContent", body

    async def _request(self, clients: Clients, method: str, url: str, **kwargs) -> dict:
        """发送请求并返回响应 JSON（只重试确定没有被处理的失败）。中转服务的报错可能原样带上 API Key，展示前打码。"""
        key = self.api_key
        try:
            resp = await with_retries(lambda: send(
                clients.http, method, url, label=self.label, proxy=self.proxy, on_error=gemini_error,
                headers=self.headers, **kwargs,
            ))
            data = json_body(resp, self.label)
            error = data.get("error")
            if isinstance(error, dict) and error and not (data.get("candidates") or data.get("steps")):
                # 部分中转服务出错时也返回 HTTP 200，错误码按响应体里的为准
                code = error.get("code")
                status = code if isinstance(code, int) and 400 <= code < 600 else resp.status_code
                raise gemini_error(self.label, status, data, resp.headers)
            return data
        except DrawError as e:
            if len(key) >= 8 and key in str(e):
                e.args = (str(e).replace(key, mask_secret(key)),)
            raise

    async def _call(self, clients: Clients, api: str, prompt: str, ratio: str, tier: str, seed: int | None,
                    extra: dict, images: list[tuple[str, str]] | tuple = ()) -> tuple[bytes, str, dict]:
        url, body = self.build_request(api, prompt, ratio, tier, seed, images)
        data = await self._request(clients, "POST", url, timeout=self.timeout, json=merge_extra(body, extra))
        usage = gemini_usage(data)
        if api != "interactions":
            b64, text = parse_generate_content(data, edit=bool(images))
            return decode_image(b64), text, usage
        block, text = parse_interaction(data)
        if block.get("data"):
            return decode_image(block["data"]), text, usage
        # 只有请求 delivery=uri 时才会返回链接（本插件不请求）；API Key 只发给 API 地址本身（协议、主机、端口都相同）
        uri = as_str(block.get("uri"))
        same_origin = urlparse(uri)[:2] == urlparse(self.base_url)[:2]
        image = await download_image(clients.http, uri, label=self.label, proxy=self.proxy, timeout=self.timeout,
                                     headers=self.headers if same_origin else None)
        return image, text, usage

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        if self.missing():
            raise DrawError(self.missing())
        ratio, tier = self.image_config(req.orientation)
        images: list[tuple[str, str]] = []
        dropped = 0
        if req.is_edit:
            reason = self.edit_missing()
            if reason:
                raise DrawError(reason)
            limit = self.edit_limit
            dropped = max(0, len(req.images) - limit)
            images = encode_inputs(req.images[:limit], self.label)  # 先校验，格式或大小不对时不发请求
            ratio = self.edit_ratio(req, ratio)
        seed = as_int(req.seed, 0)
        seed = seed % 2**31 if seed else None  # seed 是 int32
        prompt = final_prompt(req, self.supports_negative)
        extra = self.extra_body()  # 先解析，JSON 有误时不发请求
        api = "interactions" if self.api == "interactions" else "generate_content"
        try:
            image, text, usage = await self._call(clients, api, prompt, ratio, tier, seed, extra, images)
        except DrawError as e:
            # 自动模式：generateContent 不提供该模型（404，不计费）时改用 Interactions API 再试一次。
            # 额外请求参数按 generateContent 的格式填写，回退时不附加。
            if self.api != "auto" or not _not_served(e):
                raise
            api = "interactions"
            try:
                image, text, usage = await self._call(clients, api, prompt, ratio, tier, seed, {}, images)
            except DrawError as e2:
                if e2.status == 404:  # 两个接口都找不到，多半是模型名或 API 地址有误，报告第一个错误
                    raise e from e2
                raise
        params = {"model": self.model, "size": size_text(ratio, tier), "aspect_ratio": ratio or "auto",
                  "api": "Interactions" if api == "interactions" else "generateContent"}
        if images:
            params.update(mode="edit", input_images=len(images))
            if dropped:  # 超出当前模型的输入图片数，只使用了前几张
                params["input_images_dropped"] = dropped
        if tier:
            params["image_size"] = tier
        if self.thinking_level:
            params["thinking_level"] = self.thinking_level
        if seed is not None:
            params["seed"] = seed
        params["usage"] = usage
        return image_result(image, params, text)

    async def list_models(self, clients: Clients) -> list[str]:
        """列出名称含 image / banana 且支持 generateContent 的模型。"""
        if self.missing():
            raise DrawError(self.missing())
        names, token = [], ""
        for _ in range(20):  # 防止分页异常导致死循环
            params = {"pageSize": 1000, **({"pageToken": token} if token else {})}
            data = await self._request(clients, "GET", f"{self.base_url}/models", timeout=min(self.timeout, 30.0),
                                       params=params)
            for item in _list(data.get("models")):
                if not isinstance(item, dict):
                    continue
                name = as_str(item.get("name"))
                name = name[7:] if name.startswith("models/") else name
                methods = item.get("supportedGenerationMethods")
                if isinstance(methods, list) and "generateContent" not in methods:
                    continue  # 如只支持 predict 的 Imagen
                looks = f"{name} {as_str(item.get('displayName'))}".lower()
                if name and name not in names and ("image" in looks or "banana" in looks) \
                        and not name.lower().startswith("imagen"):
                    names.append(name)
            token = as_str(data.get("nextPageToken"))
            if not token:
                break
        return names

    def summary(self) -> str:
        ratio, tier = self.image_config("portrait")
        model = self.model
        if model != self.configured_model:
            model += f"（{self.configured_model} 已停用）"
        return f"Gemini / {model} / 竖图 {ratio}{' ' + tier if tier else ''}"
