"""SenseNova（商汤日日新）U1.5 文生图后端。

文档：https://platform.sensenova.cn/docs#model-u1-5 （国际站 https://platform.sensenova.ai/docs）。
POST {base}/v1/images/generations，JSON 请求体，同步返回图片；认证为 Authorization: Bearer sk-...。
接口没有负面提示词、种子字段，每次只能生成一张。
"""

import math
import re
from urllib.parse import urlsplit, urlunsplit

from .base import (
    Backend,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    as_bool,
    as_float,
    as_str,
    decode_b64_image,
    download_image,
    error_from_response,
    final_prompt,
    image_result,
    json_body,
    normalize_base_url,
    parse_extra_json,
    parse_ratio,
    parse_size,
    response_detail,
    send,
    with_retries,
)

DEFAULT_BASE_URL = "https://token.sensenova.cn/v1"  # 国际站：https://token.sensenova.ai/v1（Key 按站点区分）
DEFAULT_MODEL = "sensenova-u1.5-lite"
# 官方文档目前只列出这两个绘图模型
KNOWN_MODELS = ("sensenova-u1.5-lite", "sensenova-u1.5-fast")
# 官方客户端只允许 U1.5 Lite 使用 4K，其他模型填 4K 时改用 2K
FOUR_K_MODELS = ("sensenova-u1.5-lite",)
# 2K 为文档推荐尺寸，1K 取自官方客户端，4K 为文档的 4096x4096 及同规则的 2:3 / 3:2；均为 32 的倍数
PRESETS = {
    "1K": {"portrait": "1088x1632", "landscape": "1632x1088", "square": "1344x1344"},
    "2K": {"portrait": "1664x2496", "landscape": "2496x1664", "square": "2048x2048"},
    "4K": {"portrait": "2720x4096", "landscape": "4096x2720", "square": "4096x4096"},
}
DEFAULT_SIZES = PRESETS["2K"]
OUTPUT_FORMATS = ("png", "jpeg", "webp")
DEFAULT_TIMEOUT = 300.0  # 同步生成 2K / 4K 较慢，官方客户端同样使用 300 秒
# 文档的尺寸限制：宽高为 32 的倍数、512-4096、宽高比不超过 3:1
MIN_SIDE, MAX_SIDE, MAX_RATIO = 512, 4096, 3

_SAFETY_RE = re.compile(r"safe|sensitive|moderat|censor|violat|审核|安全|敏感|违规|违禁|风险", re.I)
_RATE_RE = re.compile(r"\brate\b|rate[ _-]?limit|too many|frequen|concurren|\bqps\b|\brpm\b|频繁|频率|速率|并发", re.I)
_QUOTA_RE = re.compile(r"quota|\bpoints?\b|credit|balance|insufficient|exhaust|积分|额度|余额|欠费|用尽|用完|不足", re.I)


# ---------------------------------------------------------------- 尺寸


def _round32(value: float) -> int:
    return min(MAX_SIDE, max(MIN_SIDE, round(value / 32) * 32))


def snap_size(width: float, height: float) -> str:
    """把任意宽高调整到文档允许的范围内，返回 "宽x高"。"""
    w, h = float(width), float(height)
    # 先把宽高比限制在 3:1 以内（加长短边），再整体缩放到 512-4096 之间
    w, h = max(w, h / MAX_RATIO), max(h, w / MAX_RATIO)
    scale = min(1.0, MAX_SIDE / max(w, h))
    scale = max(scale, MIN_SIDE / min(w, h))
    w, h = _round32(w * scale), _round32(h * scale)
    while max(w, h) > MAX_RATIO * min(w, h):  # 取整后可能略超 3:1
        if w < h:
            w += 32
        else:
            h += 32
    return f"{w}x{h}"


def resolve_size(value, orientation: str, model: str) -> str:
    """把配置中的尺寸转换成接口接受的值。

    支持 "1664x2496" / "2048 x 2048" / "2048*2048"、1K / 2K / 4K 档位、宽高比 "3:4"（按 2K 面积计算）和 auto；
    其他宽高会被调整到文档限制内，无法识别时使用该方向的默认尺寸。
    """
    text = as_str(value)
    tier = text.upper()
    if tier == "AUTO":
        return "auto"
    if tier in PRESETS:
        if tier == "4K" and model.lower() not in FOUR_K_MODELS:
            tier = "2K"
        return PRESETS[tier][orientation]
    width, height = parse_size(text, (0, 0))
    if width and height:
        return snap_size(width, height)
    ratio = parse_ratio(text)
    if ratio:
        r, area = ratio[0] / ratio[1], 2048 * 2048
        return snap_size(math.sqrt(area * r), math.sqrt(area / r))
    return DEFAULT_SIZES[orientation]


def _flag(value, default: bool) -> bool:
    """开关配置：空值与无法识别的字符串使用默认值（bool("false") 为 True，不能直接用）。"""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return default
    return as_bool(value, default)


# ---------------------------------------------------------------- 错误


def map_error(label: str, status: int, body, headers) -> DrawError:
    """按 SenseNova 的错误格式生成说明。

    文档格式为 {"error": {"type", "code", "message"}}；网关的 401 / 404 没有 type，code 为整数，
    或只有顶层 message。429 只在频率限制时重试一次，积分额度用尽不重试；5xx 沿用默认规则。
    内容审核拒绝按文档是 400 failed_precondition_error 或 403，参数错误（invalid_request_error）不按审核处理。
    """
    err = error_from_response(label, status, body, headers)
    detail = response_detail(body)
    suffix = f"：{detail}" if detail else ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        kind, message = as_str(error.get("type")), as_str(error.get("message"))
    else:
        kind, message = "", detail
    if status == 401:
        return DrawError(
            f"{label} API Key 无效或缺失（HTTP 401）{suffix}。Key 在 SenseNova 控制台的 API Keys 页面创建，"
            "国际站的 Key 需要把 API 地址改成 https://token.sensenova.ai/v1", status=status)
    if status == 429:
        if _QUOTA_RE.search(message) and not _RATE_RE.search(message):
            return DrawError(f"{label} 积分额度已用尽（HTTP 429）：额度按滚动 5 小时和每周计算，请稍后再试或到控制台查看余额"
                             f"{suffix}", status=status)
        return DrawError(f"{label} 请求过于频繁（HTTP 429），请稍后再试{suffix}", status=status, retryable=True,
                         retry_after=err.retry_after)
    if status < 500 and kind != "invalid_request_error" and _SAFETY_RE.search(message):
        return DrawError(f"{label} 拒绝生成：提示词或画面未通过内容安全检查，请修改描述后再试（HTTP {status}）{suffix}",
                         status=status)
    if kind == "failed_precondition_error":
        # 文档：编码失败、引擎不可用、安全检查未通过。只有明确是引擎不可用时才重试一次
        return DrawError(f"{label} 拒绝生成（HTTP {status}）：可能未通过内容安全检查，或编码失败、引擎暂不可用{suffix}",
                         status=status, retryable=bool(re.search(r"engine|引擎", message, re.I)))
    if status == 403:
        return DrawError(f"{label} 拒绝了请求（HTTP 403）：提示词语言不受支持、被内容策略拦截，或 Key 无权使用该模型{suffix}",
                         status=status)
    if status == 404:
        return DrawError(f"{label} 接口地址或模型不存在（HTTP 404）：请检查 API 地址（默认 {DEFAULT_BASE_URL}）"
                         f"和模型名（如 {DEFAULT_MODEL}）{suffix}", status=status)
    return err


# ---------------------------------------------------------------- 后端


class SenseNovaBackend(Backend):
    key = "sensenova"
    label = "SenseNova"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    doc_url = "https://platform.sensenova.cn/docs#model-u1-5"

    @property
    def api_key(self) -> str:
        return re.sub(r"^bearer\s+", "", as_str(self.section.get("api_key")), flags=re.I)

    @property
    def model(self) -> str:
        return as_str(self.section.get("model")) or DEFAULT_MODEL

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    @property
    def output_format(self) -> str:
        fmt = as_str(self.section.get("output_format")).lower()
        fmt = "jpeg" if fmt == "jpg" else fmt  # 文生图接口只接受 jpeg
        return fmt if fmt in OUTPUT_FORMATS else "png"

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def size(self, orientation: str) -> str:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        return resolve_size(self.section.get(f"{orientation}_size"), orientation, self.model)

    def url(self, path: str) -> str:
        """截掉第一个 /v1 之后的路径（用户可能粘贴了完整接口地址），没有 /v1 时补上，再拼接接口路径。"""
        try:
            parts = urlsplit(normalize_base_url(as_str(self.section.get("base_url")), DEFAULT_BASE_URL))
        except ValueError as e:
            raise DrawError(f"{self.label} API 地址格式错误：{e}") from e
        match = re.search(r"/v1(?=/|$)", parts.path, re.I)
        prefix = parts.path[:match.end()] if match else parts.path.rstrip("/") + "/v1"
        return urlunsplit((parts.scheme, parts.netloc, prefix + path, "", ""))

    def missing(self) -> str:
        return "" if self.api_key else "管理员还没有配置 SenseNova API Key"

    def build_body(self, req: DrawRequest) -> dict:
        # n 只能为 1（默认值，不发送）；接口没有 negative_prompt / seed，负面内容写进提示词
        body = {
            "model": self.model,
            "prompt": final_prompt(req, self.supports_negative),
            "size": self.size(req.orientation),
            "response_format": "b64_json",
            "output_format": self.output_format,
            # 接口默认加水印，文档建议总是明确发送
            "watermark": _flag(self.section.get("watermark"), False),
            "prompt_extend": _flag(self.section.get("prompt_extend"), True),
        }
        extra = self.section.get("extra_body")
        body.update(extra if isinstance(extra, dict) else parse_extra_json(as_str(extra), self.label))
        return body

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        if missing := self.missing():
            raise DrawError(missing)
        body = self.build_body(req)
        url = self.url("/images/generations")
        resp = await with_retries(lambda: send(
            clients.http, "POST", url, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=map_error, headers=self.headers, json=body,
        ))
        data = json_body(resp, self.label)
        items = data.get("data")
        item = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
        if item.get("b64_json"):
            image = decode_b64_image(item["b64_json"], self.label)
        elif item.get("url"):
            # response_format=url 时返回 24 小时有效的临时链接，下载时不带 Key
            image = await download_image(clients.http, item["url"], label=self.label, proxy=self.proxy,
                                         timeout=min(self.timeout, 120))
        elif data.get("error"):
            # HTTP 200 但响应体是错误对象；已经返回图片时以图片为准（可能已扣费）
            raise map_error(self.label, resp.status_code, data, resp.headers)
        else:
            raise DrawError(f"{self.label} 没有返回图片（可能未通过内容安全检查）")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        params = {
            "model": body.get("model"),
            # 响应中的 size 为实际输出尺寸（请求 auto 时尤其有用）
            "size": as_str(data.get("size")) or body.get("size"),
            "output_format": body.get("output_format"),
            "watermark": body.get("watermark"),
            "prompt_extend": body.get("prompt_extend"),
        }
        if usage.get("total_tokens") is not None:
            params["tokens"] = usage["total_tokens"]
        return image_result(image, params)

    async def list_models(self, clients: Clients) -> list[str]:
        """GET /v1/models，只保留能输出图片的模型；列表中没有时返回文档中的模型。"""
        if missing := self.missing():
            raise DrawError(missing)
        url = self.url("/models")
        resp = await with_retries(lambda: send(
            clients.http, "GET", url, label=self.label, proxy=self.proxy, timeout=min(self.timeout, 30),
            on_error=map_error, headers=self.headers,
        ))
        items = json_body(resp, self.label).get("data")
        models = []
        for item in items if isinstance(items, list) else []:
            model_id = as_str(item.get("id")) if isinstance(item, dict) else ""
            outputs = item.get("output_modalities") if model_id else None
            is_image = isinstance(outputs, list) and "image" in outputs
            if model_id and model_id not in models and (is_image or model_id.lower().startswith("sensenova-u")):
                models.append(model_id)
        return models or list(KNOWN_MODELS)

    def summary(self) -> str:
        return f"{self.label} / {self.model} / 竖图 {self.size('portrait')}"
