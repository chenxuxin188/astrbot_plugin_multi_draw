"""xAI（Grok Imagine）文生图后端：POST {base}/images/generations。

xAI 没有像素尺寸、种子和负面提示词，只能指定宽高比（aspect_ratio）、分辨率（1k / 2k）和画质（仅 2.0 支持）。
发送 size 字段会直接返回 400（Argument not supported: size），宽高比 / 分辨率写错会返回 422，
所以配置中的值都会先换成接口接受的值再发送。
"""

import math
import re
from urllib.parse import urlsplit

from .base import (
    MAX_DIMENSION,
    Backend,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    as_float,
    as_str,
    decode_b64_image,
    download_image,
    error_from_response,
    final_prompt,
    image_result,
    json_body,
    mask_url_credentials,
    normalize_base_url,
    parse_extra_json,
    response_detail,
    send,
    with_retries,
)

DEFAULT_BASE_URL = "https://api.x.ai/v1"  # 全局地址；us.api.x.ai 等区域地址不提供图片接口
DEFAULT_MODEL = "grok-imagine-image-2.0"
DEFAULT_SIZES = {"portrait": "2:3", "landscape": "3:2", "square": "1:1"}
DEFAULT_RESOLUTION = "auto"
DEFAULT_QUALITY = "medium"

# 接口接受的宽高比（另有 auto：由模型决定，此时不发送该字段）
RATIOS = ("1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1",
          "21:9", "5:2")
# 21:9 与 5:2 是 grok-imagine-image-2.0 新增的比例，其他模型保守起见换成最接近的比例
V1_RATIOS = RATIOS[:-2]
# 接口枚举里还有 1.5k，但官方文档只列出 1k（约 100 万像素）和 2k（约 400 万像素）；值区分大小写
RESOLUTIONS = ("1k", "2k")
QUALITIES = ("low", "medium", "auto")  # 仅 grok-imagine-image-2.0 支持；auto 目前等同 low

# 用户常把对话等接口的完整地址粘贴为 API 地址，去掉这些后缀
_URL_SUFFIXES = ("/chat/completions", "/completions", "/responses", "/messages", "/images/generations",
                 "/images/edits", "/images", "/image-generation-models", "/models")
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([:x×*/])\s*(\d+(?:\.\d+)?)")
_MODERATION_WORDS = ("moderation", "content policy", "safety", "violat", "inappropriate")
# 额度用尽、达到每月消费上限时 xAI 也可能返回 429，这种情况重试没有意义
_BILLING_WORDS = ("credit", "spending limit", "billing", "quota", "balance", "purchase")


def model_family(model: str) -> str:
    """v2：grok-imagine-image-2.0 系列；legacy：已下线的 grok-2-image；其他按 1.0 处理。"""
    name = model.lower()
    # 注意 1.0 的别名 grok-imagine-image-2026-03-02 也以 grok-imagine-image-2 开头，要带上小数点判断
    if "grok-imagine-image-2." in name:
        return "v2"
    if "grok-2-image" in name:
        return "legacy"
    return "v1"


def _ratio_value(ratio: str) -> float:
    width, height = ratio.split(":")
    return float(width) / float(height)


def nearest_ratio(width: float, height: float, allowed: tuple[str, ...]) -> str:
    """取最接近的可用宽高比（按对数距离）。不能约分，例如 832x1216 约分成 13:19 会被接口拒绝。"""
    target = math.log(width / height)
    return min(allowed, key=lambda r: abs(math.log(_ratio_value(r)) - target))


def parse_aspect(text, allowed: tuple[str, ...], default: str) -> tuple[str, int]:
    """把配置的尺寸换成接口接受的宽高比，返回 (宽高比, 像素数)。

    auto 返回空串（不发送，由模型决定）；填写像素尺寸（如 832x1216）时像素数非 0，用于推断分辨率；
    无法识别时使用 default。
    """
    text = as_str(text).lower().replace("：", ":")
    if text == "auto":
        return "", 0
    if text in allowed:
        return text, 0
    match = _SIZE_RE.fullmatch(text)
    width, height = (float(match[1]), float(match[3])) if match else (0.0, 0.0)
    # 数值过大时视为无法识别（避免 inf 或像素数溢出）
    if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
        return default, 0
    pixels = int(width * height) if match[2] in "x×*" else 0
    return nearest_ratio(width, height, allowed), pixels


def xai_error(label: str, status: int, body, headers) -> DrawError:
    """xAI 的错误响应：内容审核、额度用尽给出明确提示；只重试限流的 429 和服务不可用的 503。

    超时和 500 / 502 / 504 都可能已经生成并扣费，不自动重试。
    """
    detail = response_detail(body)
    lower, suffix = detail.lower(), f"：{detail}" if detail else ""
    reason = ""
    if status in (400, 403, 422) and any(word in lower for word in _MODERATION_WORDS):
        reason = "拒绝生成：内容未通过安全审核，请修改描述后重试"
    elif status in (402, 403, 429) and any(word in lower for word in _BILLING_WORDS):
        reason = "账户额度已用尽或达到消费上限，请到 https://console.x.ai 充值或调高消费上限"
    elif status == 401:
        reason = "API Key 无效或已过期，请到 https://console.x.ai 检查"
    elif status in (400, 422) and "prompt" in lower and any(w in lower for w in ("long", "length", "exceed")):
        reason = "提示词过长，请缩短描述"
    if reason:
        return DrawError(f"{label} {reason}（HTTP {status}）{suffix}", status=status)
    error = error_from_response(label, status, body, headers)
    # 限流和 503 的请求被拒绝、没有开始生成，可以安全重试
    error.retryable = status in (429, 503)
    if not error.retryable:
        error.retry_after = None
    return error


class XAIBackend(Backend):
    key = "xai"
    label = "xAI"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    max_concurrency = None
    doc_url = "https://docs.x.ai/developers/model-capabilities/images/generation"

    @property
    def api_key(self) -> str:
        """去掉误粘贴的 Bearer 前缀和空白（含中间的换行），否则请求头非法，报错信息还会带出 Key。"""
        key = as_str(self.section.get("api_key"))
        if key.lower().startswith("bearer "):
            key = key[7:]
        return "".join(key.split())

    @property
    def base_url(self) -> str:
        """API 根地址：只填域名时补 /v1，去掉误粘贴的 /images/generations、/chat/completions 等后缀；
        us.api.x.ai 等区域地址不提供绘图接口，改用全局地址。"""
        url = normalize_base_url(self.get("base_url"), DEFAULT_BASE_URL)
        try:
            parts = urlsplit(url)
            host, _ = (parts.hostname or "").lower(), parts.port  # 端口不是数字或超出范围时抛出 ValueError
        except ValueError as e:
            raise DrawError(f"xAI API 地址格式错误：{e}") from e
        if parts.scheme.lower() not in ("http", "https") or not host:
            raise DrawError(f"xAI API 地址无效，请使用 http:// 或 https:// 开头的地址：{mask_url_credentials(url)}")
        netloc = "api.x.ai" if host.endswith(".api.x.ai") else parts.netloc
        path = parts.path.rstrip("/")
        for suffix in _URL_SUFFIXES:
            if path.lower().endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                break
        return f"{parts.scheme}://{netloc}{path or '/v1'}"

    @property
    def model(self) -> str:
        return self.get("model", DEFAULT_MODEL)

    @property
    def resolution(self) -> str:
        value = self.get("resolution", DEFAULT_RESOLUTION).lower()
        return value if value in RESOLUTIONS else "auto"

    @property
    def quality(self) -> str:
        value = self.get("quality", DEFAULT_QUALITY).lower()
        return value if value in QUALITIES else "auto"  # 无效值不发送，等同 auto

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def aspect(self, orientation: str, family: str) -> tuple[str, str]:
        """返回 (宽高比, 分辨率)，宽高比为空表示 auto。"""
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        default = DEFAULT_SIZES[orientation]
        allowed = RATIOS if family == "v2" else V1_RATIOS
        ratio, pixels = parse_aspect(self.get(f"{orientation}_size", default), allowed, default)
        resolution = self.resolution
        if resolution == "auto":
            # 1k 约 100 万像素、2k 约 400 万像素，按几何中点 200 万像素划分
            resolution = "2k" if pixels > 2_000_000 else "1k"
        return ratio, resolution

    def plan(self, req: DrawRequest) -> dict:
        ratio, resolution = self.aspect(req.orientation, model_family(self.model))
        return {"model": self.model, "size": f"{ratio or 'auto'} {resolution}".strip()}

    def missing(self) -> str:
        key = self.api_key
        if not key:
            return "管理员还没有配置 xAI API Key"
        if not (key.isascii() and key.isprintable()):
            return "xAI API Key 含有中文、全角或不可见字符，请重新复制"
        try:
            self.base_url
        except DrawError as e:
            return str(e)
        return ""

    def _require(self) -> None:
        reason = self.missing()
        if reason:
            raise DrawError(reason)

    def build_body(self, req: DrawRequest) -> dict:
        model = self.model
        family = model_family(model)
        body = {"model": model, "prompt": final_prompt(req, self.supports_negative), "n": 1,
                "response_format": "b64_json"}
        # grok-2-image 只支持 model / prompt / n / response_format
        if family != "legacy":
            ratio, resolution = self.aspect(req.orientation, family)
            if ratio:
                body["aspect_ratio"] = ratio
            body["resolution"] = resolution
            if family == "v2" and self.quality != "auto":
                body["quality"] = self.quality
        extra = self.section.get("extra_body")
        body.update(extra if isinstance(extra, dict) else parse_extra_json(as_str(extra), self.label))
        return body

    @staticmethod
    def params_of(body: dict) -> dict:
        """按实际发送的请求体（含额外参数）整理写入记录的参数。"""
        ratio = as_str(body.get("aspect_ratio")) or "auto"
        resolution = as_str(body.get("resolution"))
        params = {"model": as_str(body.get("model")), "size": f"{ratio} {resolution}".strip(), "aspect_ratio": ratio}
        if resolution:
            params["resolution"] = resolution
        if body.get("quality"):
            params["quality"] = as_str(body["quality"])
        return params

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        self._require()
        body = self.build_body(req)
        url = f"{self.base_url}/images/generations"
        resp = await with_retries(lambda: send(
            clients.http, "POST", url, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=xai_error, headers=self.headers, json=body,
        ))
        data = json_body(resp, self.label)
        items = data.get("data")
        item = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else None
        if item is None:
            detail = response_detail(data) if data.get("error") else ""
            raise DrawError(f"xAI 没有返回图片{'：' + detail if detail else ''}")
        b64, image_url = as_str(item.get("b64_json")), as_str(item.get("url"))
        if item.get("respect_moderation") is False or not (b64 or image_url):
            raise DrawError("xAI 拒绝生成：生成的图片未通过内容审核，请修改描述后重试")
        if b64:
            image = decode_b64_image(b64, self.label)
        else:
            # 临时链接，需要立即下载；不带 API Key
            image = await download_image(clients.http, image_url, label=self.label, proxy=self.proxy,
                                         timeout=self.timeout)
        params = self.params_of(body)
        usage = data.get("usage")
        ticks = as_float(usage.get("cost_in_usd_ticks"), -1) if isinstance(usage, dict) else -1
        if ticks >= 0:
            params["cost_usd"] = round(ticks / 1e10, 4)  # 1 美元 = 100 亿 ticks
        # revised_prompt 已不再返回，兼容旧接口和中转服务
        return image_result(image, params, as_str(item.get("revised_prompt")))

    async def list_models(self, clients: Clients) -> list[str]:
        self._require()
        base_url = self.base_url

        async def fetch(path: str) -> dict:
            resp = await with_retries(lambda: send(
                clients.http, "GET", f"{base_url}{path}", label=self.label, proxy=self.proxy, timeout=30,
                headers=self.headers,
            ))
            return json_body(resp, self.label)

        try:
            data = await fetch("/image-generation-models")
            items = data.get("models") or data.get("data")
            image_only = False
        except DrawError as e:
            if e.status != 404:
                raise
            # 部分中转服务只实现了 /models，按名称筛选绘图模型
            data = await fetch("/models")
            items = data.get("data") or data.get("models")
            image_only = True
        models = []
        for item in items if isinstance(items, list) else []:
            model_id = as_str(item.get("id")) if isinstance(item, dict) else ""
            if model_id and model_id not in models and (not image_only or "image" in model_id.lower()):
                models.append(model_id)
        return models

    def summary(self) -> str:
        family = model_family(self.model)
        if family == "legacy":
            return f"xAI / {self.model} / 竖图 auto"
        ratio, resolution = self.aspect("portrait", family)
        text = f"xAI / {self.model} / 竖图 {ratio or 'auto'} {resolution}"
        if family == "v2" and self.quality != "auto":
            text += f" / 画质 {self.quality}"
        return text
