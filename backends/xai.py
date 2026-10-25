"""xAI（Grok Imagine）后端：文生图 POST {base}/images/generations，修改图片 POST {base}/images/edits，
余额通过管理 API（https://management-api.x.ai，需要单独的管理密钥）查询。

xAI 没有像素尺寸、种子和负面提示词，只能指定宽高比（aspect_ratio）、分辨率（1k / 2k）和画质（仅 2.0 支持）。
发送 size 字段会直接返回 400（Argument not supported: size），宽高比 / 分辨率写错会返回 422，
所以配置中的值都会先换成接口接受的值再发送。

修改图片只接受 JSON（multipart 会返回 415）：一张图片放在 image，多张放在 images（两者不能同时出现），
图片以 base64 data URL 发送；不发送宽高比时输出沿用第一张图片的比例。没有重绘幅度之类的参数。
"""

import base64
import math
import re
import uuid
from decimal import Decimal, InvalidOperation
from functools import partial
from urllib.parse import quote, urlsplit

from .base import (
    MAX_DIMENSION,
    Backend,
    BalanceInfo,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    InputImage,
    as_float,
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
    response_detail,
    send,
    usage_of,
    with_retries,
)

DEFAULT_BASE_URL = "https://api.x.ai/v1"  # 全局地址；us.api.x.ai 等区域地址不提供图片接口
DEFAULT_MODEL = "grok-imagine-image-2.0"
DEFAULT_SIZES = {"portrait": "2:3", "landscape": "3:2", "square": "1:1"}
DEFAULT_RESOLUTION = "auto"
DEFAULT_QUALITY = "medium"

# 修改图片：输入图片只能是 JPEG / PNG / WebP（按文件头判断，不相信调用方给的 MIME）
EDIT_MIMES = ("image/jpeg", "image/png", "image/webp")
# 每次修改最多使用的输入图片数：2.0 官方文档为 5 张；1.0（含 -quality / -pro）没有写明，按以前的 3 张；
# grok-2-image 不支持修改图片
EDIT_MAX_BY_FAMILY = {"v2": 5, "v1": 3, "legacy": 0}
# 网关对整个 JSON 请求体的上限约 50 MiB（实测 49.5 MiB 可以解析、50.5 MiB 返回 413），图片总大小留出余量
MAX_EDIT_BODY = 45 * 1024 * 1024
# 修改指令中的 "image 1"、"image 2"：多图时官方要求写成 <IMAGE_0>、<IMAGE_1>
_IMAGE_REF_RE = re.compile(r"\bimage\s*#?\s*(\d+)\b", re.I)
# extra_body 中这些字段在修改图片时忽略：image / images 由插件填写，size 会被接口拒绝
_EDIT_RESERVED = ("image", "images", "size")

# 余额：管理 API 与管理密钥（与 API Key 不同，在控制台 Settings → Management Keys 创建）
MANAGEMENT_BASE_URL = "https://management-api.x.ai"
BILLING_URL = "https://console.x.ai/team/default/billing"

# 接口接受的宽高比（另有 auto：由模型决定，此时不发送该字段）
RATIOS = ("1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1",
          "21:9", "5:2")
# 21:9 与 5:2 是 grok-imagine-image-2.0 新增的比例，其他模型保守起见换成最接近的比例
V1_RATIOS = RATIOS[:-2]
# 接口枚举里还有 1.5k，但官方文档只列出 1k（约 100 万像素）和 2k（约 400 万像素）；值区分大小写
RESOLUTIONS = ("1k", "2k")
# 仅 grok-imagine-image-2.0 支持；auto（不发送）时文生图目前按 low、修改图片按 medium 生成和计费
QUALITIES = ("low", "medium", "auto")

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


def xai_error(label: str, status: int, body, headers, *, edit: bool = False) -> DrawError:
    """xAI 的错误响应：内容审核、额度用尽给出明确提示；只重试限流的 429 和服务不可用的 503。

    超时和 500 / 502 / 504 都可能已经生成并扣费，不自动重试（修改图片按输入和输出图片一起计费）。
    """
    detail = response_detail(body)
    lower, suffix = detail.lower(), f"：{detail}" if detail else ""
    reason = ""
    if status in (400, 403, 422) and any(word in lower for word in _MODERATION_WORDS):
        reason = ("拒绝修改：图片或描述未通过安全审核，请修改描述或换一张图片后重试" if edit
                  else "拒绝生成：内容未通过安全审核，请修改描述后重试")
    elif status in (402, 403, 429) and any(word in lower for word in _BILLING_WORDS):
        reason = "账户额度已用尽或达到消费上限，请到 https://console.x.ai 充值或调高消费上限"
    elif status == 401 or (status == 400 and "incorrect api key" in lower):
        # Key 填错时 xAI 返回的是 400 {"code":"invalid-argument","error":"Incorrect API key provided..."}
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


# ---------------------------------------------------------------- 修改图片


def edit_limit(model: str) -> int:
    """当前模型修改图片最多使用的输入图片数，0 表示不支持修改图片。"""
    return EDIT_MAX_BY_FAMILY.get(model_family(model), 0)


class _EditLimit:
    """max_edit_images：在类上读取时是所有模型中的最大值（2.0 的 5 张），在实例上按当前模型返回。

    插件和设置页面都按实例读取，收集图片时就不会超过当前模型的上限。
    """

    def __get__(self, obj, owner=None) -> int:
        if obj is None:
            return max(EDIT_MAX_BY_FAMILY.values())
        return edit_limit(obj.model)


def edit_data_url(image: InputImage, index: int = 1) -> str:
    """把输入图片转成 base64 data URL；MIME 按文件头判断。"""
    mime = detect_mime(image.data)
    if mime not in EDIT_MIMES:
        kind = f"{mime.split('/')[-1].upper()} 格式" if mime else "无法识别的格式"
        raise DrawError(f"xAI 修改图片只支持 JPEG、PNG、WebP 格式，第 {index} 张图片是 {kind}")
    return f"data:{mime};base64,{base64.b64encode(image.data).decode()}"


def tag_image_refs(prompt: str, count: int) -> str:
    """多图修改时官方要求用 <IMAGE_0>、<IMAGE_1> 指代各张图片；插件的修改指令写的是 image 1、image 2（从 1 开始）。"""
    if count < 2:
        return prompt

    def repl(match: re.Match) -> str:
        number = int(match[1])
        return f"<IMAGE_{number - 1}>" if 1 <= number <= count else match[0]

    return _IMAGE_REF_RE.sub(repl, prompt)


# ---------------------------------------------------------------- 余额


def usd_cents(value) -> Decimal | None:
    """管理 API 的金额是 {"val": "<整数字符串>"}，单位为美分；部分字段直接是字符串。无法解析时返回 None。"""
    if isinstance(value, dict):
        value = value.get("val")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def cents_text(cents: Decimal) -> str:
    return f"{Decimal(cents) / 100:,.2f} 美元"


def management_error(label: str, status: int, body, headers) -> DrawError:
    """管理 API 的错误（gRPC 网关格式 {"code": 16, "message": "...", "details": []}）。"""
    detail = response_detail(body)
    suffix = f"：{detail}" if detail else ""
    if status == 401:
        return DrawError(f"{label} 管理密钥无效（管理密钥不是 API Key，需要在控制台 Settings → Management Keys "
                         f"单独创建）（HTTP 401）{suffix}", status=status)
    if status == 403:
        return DrawError(f"{label} 管理密钥没有读取该团队账单的权限：请在 xAI 控制台为它勾选账单读取权限，"
                         f"并确认它与 API Key 属于同一个团队（HTTP 403）{suffix}", status=status)
    if status == 400 and "uuid" in detail.lower():
        return DrawError(f"{label} 团队 ID 格式不正确（HTTP 400）{suffix}", status=status)
    if status == 404:
        # 通用的 404 说明是「检查 API 地址和模型名」，管理 API 的地址是固定的，这里只可能是团队不存在
        return DrawError(f"{label} 找不到该团队的账单：请检查团队 ID，或确认管理密钥属于该团队（HTTP 404）{suffix}",
                         status=status)
    return error_from_response(label, status, body, headers)


class XAIBackend(Backend):
    key = "xai"
    label = "xAI"
    default_prompt_mode = "natural"
    supports_negative = False
    supports_models = True
    # 按当前模型：2.0 为 5 张，1.0 为 3 张，grok-2-image 为 0（在类上读取时为 5）
    max_edit_images = _EditLimit()
    supports_strength = False  # 没有重绘幅度 / 保真度参数，修改程度只能写在指令里
    supports_balance = True
    max_concurrency = None
    doc_url = "https://docs.x.ai/developers/model-capabilities/images/generation"
    billing_url = BILLING_URL

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

    @property
    def management_key(self) -> str:
        """管理密钥，只用于查询余额；同样去掉 Bearer 前缀和空白。"""
        key = as_str(self.section.get("management_key"))
        if key.lower().startswith("bearer "):
            key = key[7:]
        return "".join(key.split())

    @property
    def team_id(self) -> str:
        return as_str(self.section.get("team_id"))

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
        family = model_family(self.model)
        if req.is_edit and family != "legacy":
            ratio = self.edit_aspect(req, family)
            return {"model": self.model, "size": f"{ratio or '原图比例'} {self.edit_resolution(req.images[0])}"}
        ratio, resolution = self.aspect(req.orientation, family)
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

    def edit_missing(self) -> str:
        if not self.max_edit_images:
            return f"当前模型 {self.model} 不支持修改图片，请改用 grok-imagine-image-2.0 或 grok-imagine-image"
        return ""

    def balance_missing(self) -> str:
        key = self.management_key
        if not key:
            return ("查询 xAI 余额需要在配置中填写管理密钥（Management Key，不是 API Key）：在 https://console.x.ai 的 "
                    f"Settings → Management Keys 创建，只勾选账单读取权限；也可以到 {BILLING_URL} 查看")
        if not (key.isascii() and key.isprintable()):
            return "xAI 管理密钥含有中文、全角或不可见字符，请重新复制"
        if self.team_id and not self._team_uuid(self.team_id):
            return "xAI 团队 ID 格式不正确，应为 UUID（形如 xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx），不确定时留空即可"
        return ""  # 团队 ID 留空时在查询时自动获取

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

    # ---------------------------------------------------------------- 修改图片

    def edit_aspect(self, req: DrawRequest, family: str) -> str:
        """修改图片的宽高比：默认不发送，输出沿用第一张图片的比例；
        只有要求的方向与第一张图片不同时才发送配置中该方向的比例（从不发送 auto）。"""
        first = image_orientation(req.images[0]) if req.images else ""
        if not first or req.orientation not in DEFAULT_SIZES or req.orientation == first:
            return ""
        ratio, _ = self.aspect(req.orientation, family)
        return ratio

    def edit_resolution(self, image: InputImage) -> str:
        """修改图片的分辨率：设置为自动时按第一张图片的像素数，超过约 200 万像素用 2k，否则 1k。"""
        resolution = self.resolution
        if resolution == "auto":
            resolution = "2k" if image.width * image.height > 2_000_000 else "1k"
        return resolution

    def build_edit_body(self, req: DrawRequest) -> dict:
        model = self.model
        family = model_family(model)
        limit = edit_limit(model)
        if not limit:
            raise DrawError(self.edit_missing())
        if not req.images:
            raise DrawError("没有需要修改的图片")
        if len(req.images) > limit:
            raise DrawError(f"xAI 的 {model} 修改图片最多使用 {limit} 张图片，"
                            f"当前有 {len(req.images)} 张，请减少图片数量")
        urls = [edit_data_url(image, index) for index, image in enumerate(req.images, 1)]
        if sum(len(url) for url in urls) > MAX_EDIT_BODY:
            raise DrawError("xAI 修改图片的输入图片总大小过大（请求上限约 50 MiB），请减少图片数量或使用更小的图片")
        prompt = tag_image_refs(final_prompt(req, self.supports_negative), len(urls))
        body = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
        ratio = self.edit_aspect(req, family)
        if ratio:
            body["aspect_ratio"] = ratio
        body["resolution"] = self.edit_resolution(req.images[0])
        if family == "v2" and self.quality != "auto":
            body["quality"] = self.quality  # 不发送时修改图片按 medium 生成和计费
        extra = self.section.get("extra_body")
        extra = dict(extra) if isinstance(extra, dict) else parse_extra_json(as_str(extra), self.label)
        body.update({k: v for k, v in extra.items() if k not in _EDIT_RESERVED})
        # image 与 images 只能出现一个：一张图片用 image，多张用 images
        if len(urls) == 1:
            body["image"] = {"url": urls[0]}
        else:
            body["images"] = [{"url": url} for url in urls]
        return body

    async def _edit(self, clients: Clients, req: DrawRequest) -> DrawResult:
        body = self.build_edit_body(req)
        url = f"{self.base_url}/images/edits"
        resp = await with_retries(lambda: send(
            clients.http, "POST", url, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=partial(xai_error, edit=True), headers=self.headers, json=body,
        ))
        result = await self.parse_image_response(clients, json_body(resp, self.label), body, edit=True)
        result.params["mode"] = "edit"
        result.params["input_images"] = len(req.images)
        if "aspect_ratio" not in body:
            result.params["aspect_ratio"] = "input"
            result.params["size"] = f"原图比例 {as_str(body.get('resolution'))}".strip()
        return result

    # ---------------------------------------------------------------- 生成

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        self._require()
        if req.is_edit:
            return await self._edit(clients, req)
        body = self.build_body(req)
        url = f"{self.base_url}/images/generations"
        resp = await with_retries(lambda: send(
            clients.http, "POST", url, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=xai_error, headers=self.headers, json=body,
        ))
        return await self.parse_image_response(clients, json_body(resp, self.label), body)

    async def parse_image_response(self, clients: Clients, data: dict, body: dict, *,
                                   edit: bool = False) -> DrawResult:
        """解析生成 / 修改图片的响应（两者格式相同）。"""
        items = data.get("data")
        item = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else None
        if item is None:
            detail = response_detail(data) if data.get("error") else ""
            raise DrawError(f"xAI 没有返回图片{'：' + detail if detail else ''}")
        b64, image_url = as_str(item.get("b64_json")), as_str(item.get("url"))
        if item.get("respect_moderation") is False or not (b64 or image_url):
            raise DrawError("xAI 拒绝修改：修改后的图片未通过内容审核，请修改描述或换一张图片后重试" if edit
                            else "xAI 拒绝生成：生成的图片未通过内容审核，请修改描述后重试")
        if b64:
            image = decode_b64_image(b64, self.label)
        else:
            # 临时链接，需要立即下载；不带 API Key
            image = await download_image(clients.http, image_url, label=self.label, proxy=self.proxy,
                                         timeout=self.timeout)
        params = self.params_of(body)
        usage = data.get("usage")
        ticks = as_float(usage.get("cost_in_usd_ticks"), -1) if isinstance(usage, dict) else -1
        # 1 美元 = 100 亿 ticks；修改图片按输入图片和输出图片一起计费
        params["usage"] = usage_of(len(items), cost=ticks / 1e10 if ticks >= 0 else None, currency="USD")
        # revised_prompt 已不再返回，兼容旧接口和中转服务
        return image_result(image, params, as_str(item.get("revised_prompt")))

    # ---------------------------------------------------------------- 余额

    @staticmethod
    def _team_uuid(text: str) -> str:
        """团队 ID 必须是 UUID（否则管理 API 返回 400 Invalid uuid），返回规范写法，无效时返回空串。"""
        try:
            return str(uuid.UUID(as_str(text)))
        except (ValueError, AttributeError, TypeError):
            return ""

    def _is_official(self) -> bool:
        try:
            return (urlsplit(self.base_url).hostname or "").lower() == "api.x.ai"
        except (ValueError, DrawError):
            return False

    def _mgmt_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.management_key}"}

    def _scrub(self, text: str) -> str:
        """万一服务端在错误信息里带出了密钥，替换成掩码。"""
        for secret in (self.management_key, self.api_key):
            if len(secret) >= 8:
                text = text.replace(secret, mask_secret(secret))
        return text

    async def _get_json(self, clients: Clients, url: str, headers: dict, on_error) -> dict:
        resp = await with_retries(lambda: send(
            clients.http, "GET", url, label=self.label, proxy=self.proxy, timeout=30, on_error=on_error,
            headers=headers,
        ))
        return json_body(resp, self.label)

    async def _management_team(self, clients: Clients) -> str:
        """管理密钥所属的团队：GET /auth/management-keys/validation（不需要任何权限）。"""
        info = await self._get_json(clients, f"{MANAGEMENT_BASE_URL}/auth/management-keys/validation",
                                    self._mgmt_headers(), management_error)
        if as_str(info.get("scope")) == "SCOPE_TEAM" and as_str(info.get("scopeId")):
            return as_str(info.get("scopeId"))
        # teamId 已标记为弃用，但仍是「密钥代表的团队」；组织级密钥的 scopeId 是组织 ID，不能当团队 ID
        return as_str(info.get("teamId"))

    async def resolve_team_id(self, clients: Clients) -> tuple[str, dict, str]:
        """返回 (团队 ID, /v1/api-key 的信息, 来源)：配置 > 用 API Key 查询（官方地址时）> 管理密钥所属团队。"""
        if self.team_id:
            return self._team_uuid(self.team_id) or self.team_id, {}, "config"
        if self._is_official() and not self.missing():
            try:
                info = await self._get_json(clients, f"{self.base_url}/api-key", self.headers, xai_error)
            except DrawError:
                # 这一步只是为了拿到团队 ID：API Key 填错（400 / 401）、被禁用（403）或 api.x.ai 暂时出错时
                # 管理密钥仍然可以查询，换用管理密钥所属团队
                info = {}
            team = as_str(info.get("team_id"))
            if team:
                return team, info, "api_key"
        team = await self._management_team(clients)
        if team:
            return team, {}, "management_key"
        raise DrawError("无法确定 xAI 团队 ID，请在配置中填写团队 ID（可以在 xAI 控制台的团队设置中复制）")

    async def balance(self, clients: Clients) -> BalanceInfo:
        reason = self.balance_missing()
        if reason:
            raise DrawError(reason)
        try:
            return await self._balance(clients)
        except DrawError as e:
            message = self._scrub(str(e))
            if message == str(e):
                raise
            raise DrawError(message, status=e.status, retryable=e.retryable, retry_after=e.retry_after) from None

    async def _balance(self, clients: Clients) -> BalanceInfo:
        team, key_info, source = await self.resolve_team_id(clients)
        notes: list[str] = []

        async def ledger(team_id: str) -> dict:
            return await self._get_json(
                clients, f"{MANAGEMENT_BASE_URL}/v1/billing/teams/{quote(team_id, safe='')}/prepaid/balance",
                self._mgmt_headers(), management_error)

        try:
            data = await ledger(team)
        except DrawError as e:
            # API Key 与管理密钥可能属于不同团队：改查管理密钥所属团队
            if source != "api_key" or e.status not in (403, 404):
                raise
            other = await self._management_team(clients)
            if not other or other == team:
                raise
            data, team = await ledger(other), other
            notes.append("API Key 与管理密钥不属于同一团队，显示的是管理密钥所属团队的余额")

        total = usd_cents(data.get("total"))
        if total is None:
            raise DrawError("xAI 返回的余额数据格式异常")
        items: list[tuple[str, str]] = []
        # 预付账本：充值、退款为负数，消费为正数，所以剩余预付额度 = -合计（单位美分）
        if total <= 0:
            items.append(("预付余额", cents_text(-total if total < 0 else Decimal(0))))
        else:
            items.append(("预付余额", cents_text(Decimal(0))))
            notes.append(f"预付账本合计为 +{cents_text(total)}，消费已超过已购额度")

        # 本月后付费账单和后付费上限：可选，管理密钥没有这项权限时跳过
        base = f"{MANAGEMENT_BASE_URL}/v1/billing/teams/{quote(team, safe='')}"
        try:
            preview = await self._get_json(clients, f"{base}/postpaid/invoice/preview", self._mgmt_headers(),
                                           management_error)
        except DrawError as e:
            if e.status not in (403, 404):
                notes.append(f"未能读取后付费账单（HTTP {e.status}）" if e.status else "未能读取后付费账单")
        else:
            core = preview.get("coreInvoice") if isinstance(preview.get("coreInvoice"), dict) else {}
            due = usd_cents(core.get("amountAfterVat"))
            limit = usd_cents(preview.get("effectiveSpendingLimit"))
            if due:
                items.append(("本月后付费", cents_text(due)))
            if limit is not None:
                # 后付费上限为 0（默认）时只使用预付额度，用完后请求会被拒绝
                items.append(("后付费上限", cents_text(limit) if limit else "0（只用预付额度，用完即停）"))

        if key_info.get("team_blocked") or key_info.get("api_key_blocked") or key_info.get("api_key_disabled"):
            notes.append("当前 API Key 或团队已被停用 / 封禁")
        if not self._is_official():
            notes.append("API 地址不是 api.x.ai，这里显示的是管理密钥所属 xAI 团队的余额，不是中转服务的余额")
        notes.append("账单数据可能有延迟，以控制台为准")
        return BalanceInfo(items=items, note="；".join(notes))

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
