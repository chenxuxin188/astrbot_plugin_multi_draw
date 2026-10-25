"""Seedream（火山方舟 / BytePlus ModelArk）文生图后端。

接口：POST {base}/images/generations，Authorization: Bearer <API Key>，一次生成一张，返回 b64_json。
文档（2026-10）：https://www.volcengine.com/docs/82379/1541523
"""

import math
import re
from urllib.parse import urlsplit

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
    mask_secret,
    mask_url_credentials,
    normalize_base_url,
    parse_extra_json,
    response_detail,
    send,
    with_retries,
)

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seedream-5-0-pro-260628"
DEFAULT_TIMEOUT = 300.0
DEFAULT_OUTPUT_FORMAT = "jpeg"
# 默认尺寸落在所有现有模型都接受的范围内（3,686,400 ~ 4,624,220 像素）
DEFAULT_SIZES = {"portrait": "1664x2496", "landscape": "2496x1664", "square": "2048x2048"}
ORIENTATION_RATIO = {"portrait": (2, 3), "landscape": (3, 2), "square": (1, 1)}

# 各模型系列的规则。tiers：可用的分辨率档位；pixels：宽x高的总像素范围（宽高比都限制在 1/16 ~ 16）；
# output_format / fast（提示词快速优化）：是否接受对应字段。
# 组图开关 sequential_image_generation 默认关闭（只出一张），5.0 pro / flash 不接受该字段，因此从不发送
FAMILIES = {
    "5.0-pro": {"tiers": ("1K", "1.5K", "2K"), "pixels": (921_600, 4_624_220), "output_format": True, "fast": True},
    "5.0-flash": {"tiers": ("1K", "1.5K", "2K"), "pixels": (921_600, 4_624_220), "output_format": True, "fast": False},
    "5.0-lite": {"tiers": ("2K", "3K", "4K"), "pixels": (3_686_400, 16_777_216), "output_format": True, "fast": False},
    "4.5": {"tiers": ("2K", "4K"), "pixels": (3_686_400, 16_777_216), "output_format": False, "fast": False},
    "4.0": {"tiers": ("1K", "2K", "4K"), "pixels": (921_600, 16_777_216), "output_format": False, "fast": True},
    # 接入点 ID（ep-...）或无法识别的新模型：只用所有现有模型都接受的尺寸，不发送可选字段
    "generic": {"tiers": ("2K",), "pixels": (3_686_400, 4_624_220), "output_format": False, "fast": False},
}
_FAMILY_PATTERNS = (
    (re.compile(r"seedream-5-0-pro"), "5.0-pro"),
    (re.compile(r"seedream-5-0-flash"), "5.0-flash"),
    # 5.0 lite 的 ID 是 …-5-0-260128 或 …-5-0-lite-260128；中转服务的模型名可能不带日期
    (re.compile(r"seedream-5-0(-lite)?(-\d|$)"), "5.0-lite"),
    (re.compile(r"seedream-4-5(-\d|$)"), "4.5"),
    (re.compile(r"seedream-4-0(-\d|$)"), "4.0"),
)
# 误粘贴的完整接口地址，去掉后再拼接 /images/generations
_URL_SUFFIXES = ("/images/generations", "/images", "/chat/completions", "/responses", "/contents/generations/tasks")

TIER_VALUES = {"1K": 1.0, "1.5K": 1.5, "2K": 2.0, "3K": 3.0, "4K": 4.0}
# 官方推荐的各档位宽高；横向比例（4:3、16:9、3:2）由对应竖向比例交换宽高得到
_TABLE_5 = {  # 5.0 pro / flash
    "1K": {"1:1": (1024, 1024), "3:4": (864, 1152), "9:16": (800, 1424), "2:3": (832, 1248), "21:9": (1568, 672)},
    "1.5K": {"1:1": (1536, 1536), "3:4": (1344, 1792), "9:16": (1152, 2048), "2:3": (1248, 1872),
             "21:9": (2352, 1008)},
    "2K": {"1:1": (2048, 2048), "3:4": (1776, 2368), "9:16": (1584, 2816), "2:3": (1664, 2496), "21:9": (3136, 1344)},
}
_TABLE_4 = {  # 5.0 lite / 4.5 / 4.0（1K 仅 4.0，3K 仅 5.0 lite）
    "1K": {"1:1": (1024, 1024), "3:4": (864, 1152), "9:16": (736, 1312), "2:3": (832, 1248), "21:9": (1568, 672)},
    "2K": {"1:1": (2048, 2048), "3:4": (1728, 2304), "9:16": (1600, 2848), "2:3": (1664, 2496), "21:9": (3136, 1344)},
    "3K": {"1:1": (3072, 3072), "3:4": (2592, 3456), "9:16": (2304, 4096), "2:3": (2496, 3744), "21:9": (4704, 2016)},
    "4K": {"1:1": (4096, 4096), "3:4": (3520, 4704), "9:16": (3040, 5504), "2:3": (3328, 4992), "21:9": (6240, 2656)},
}

# 数字长度有上限：超长的数字视为无法识别，避免计算溢出
_WXH_RE = re.compile(r"(?<![\d.])(\d{1,6})\s*x\s*(\d{1,6})(?!\d|\.\d)")
_RATIO_RE = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d{1,3})?)\s*:\s*(\d{1,4}(?:\.\d{1,3})?)(?!\d|\.\d)")
_TIER_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s*k")


def detect_family(model: str) -> str:
    """按模型 ID 判断模型系列（兼容火山方舟 doubao-seedream-… 与 BytePlus dola-seedream-… / seedream-…）。"""
    name = re.sub(r"[._\s]", "-", as_str(model).lower())
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "generic"


def fit_pixels(width: int, height: int, lo: int, hi: int) -> tuple[int, int]:
    """把宽高调整到总像素 [lo, hi]、宽高比 1/16 ~ 16 以内：本来合法的尺寸原样返回，
    否则尽量保持比例缩放，边长取 16 的倍数。"""
    if lo <= width * height <= hi and 1 / 16 <= width / height <= 16:
        return width, height
    ratio = min(16.0, max(1 / 16, width / height))
    stretch = max(ratio, 1 / ratio)  # 长边 / 短边
    area = min(max(width * height, lo), hi)
    short = max(16, round(math.sqrt(area / stretch) / 16) * 16)
    for side in (short, short - 16, short + 16, short - 32, short + 32):
        if side < 16:
            continue
        # 长边需满足：不短于短边、不超过短边 16 倍、乘积在像素范围内
        long_lo = -(-max(side, -(-lo // side)) // 16) * 16
        long_hi = min(16 * side, hi // side) // 16 * 16
        if long_lo <= long_hi:
            long = min(max(round(side * stretch / 16) * 16, long_lo), long_hi)
            return (long, side) if ratio >= 1 else (side, long)
    raise DrawError(f"无法把尺寸 {width}x{height} 调整到 Seedream 允许的范围")


def _ratio_key(width: float, height: float) -> str:
    if not (float(width).is_integer() and float(height).is_integer()):
        return ""
    width, height = int(width), int(height)
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def pick_size(text, orientation: str, family: str) -> str:
    """把配置中的尺寸转换成所选模型系列接受的「宽x高」。

    支持 "1664x2496"、档位 "2K"、比例 "3:4" 以及 "9:16 2K" 这样的组合：档位和比例换成官方推荐的宽高
    （只写档位时按画面方向取 2:3 / 3:2 / 1:1），模型不支持的档位换成最接近的档位，超出像素范围的宽高按比例缩放。
    无法识别时使用默认尺寸。
    """
    rules = FAMILIES.get(family) or FAMILIES["generic"]
    lo, hi = rules["pixels"]
    orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
    text = as_str(text).lower().replace("×", "x").replace("*", "x").replace("：", ":")

    match = _WXH_RE.search(text)
    if match and int(match[1]) > 0 and int(match[2]) > 0:
        return "%dx%d" % fit_pixels(int(match[1]), int(match[2]), lo, hi)
    ratio = _RATIO_RE.search(text)
    width, height = (float(ratio[1]), float(ratio[2])) if ratio else (0.0, 0.0)
    tier = _TIER_RE.search(text)
    if width <= 0 or height <= 0:
        if not tier:
            return pick_size(DEFAULT_SIZES[orientation], orientation, family)
        width, height = ORIENTATION_RATIO[orientation]

    want = float(tier[1]) if tier else 2.0
    name = min(rules["tiers"], key=lambda t: (abs(TIER_VALUES[t] - want), TIER_VALUES[t]))
    table = (_TABLE_5 if family in ("5.0-pro", "5.0-flash") else _TABLE_4)[name]
    key, swapped = _ratio_key(width, height), _ratio_key(height, width)
    if key in table:
        size = table[key]
    elif swapped in table:
        size = table[swapped][::-1]
    else:  # 不常见的比例：按该档位的面积计算
        side, scale = table["1:1"][0], math.sqrt(width / height)
        size = (max(16, round(side * scale / 16) * 16), max(16, round(side / scale / 16) * 16))
    return "%dx%d" % fit_pixels(*size, lo, hi)


def platform_model(model: str, base_url: str) -> str:
    """火山方舟与 BytePlus 的模型 ID 不同（doubao-seedream-… / dola-seedream-…、seedream-…）。
    使用两边的官方地址时自动换成对应平台的 ID；接入点 ID 和第三方中转地址不处理。"""
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:  # 地址格式错误由 endpoint / missing() 报告
        return model
    if host.endswith("bytepluses.com") and model.startswith("doubao-seedream-"):
        rest = model[len("doubao-seedream-"):]
        # BytePlus 上 4.0 / 4.5 / 5.0 lite 没有前缀，5.0 pro / flash 及之后的模型带 dola- 前缀
        legacy = re.match(r"(4-0|4-5|5-0-(lite-)?\d)", rest)
        return ("seedream-" if legacy else "dola-seedream-") + rest
    if host.endswith("volces.com") and model.startswith(("dola-seedream-", "seedream-")):
        return "doubao-seedream-" + model.split("seedream-", 1)[1]
    return model


# ---------------------------------------------------------------- 错误处理

# 内容审核拦截（不重试，提示用户换个描述）
_MODERATION = ("SensitiveContentDetected", "RiskDetection", "PolicyViolation", "PrivacyInformation", "DeepFake")
# 额度用尽或达到消费上限的 429：重试也不会成功
_QUOTA_CODES = ("QuotaExceeded", "AccountQuotaExceeded", "SetLimitExceeded", "ModelAccountTpdRateLimitExceeded")
# 按错误码前缀匹配的说明，越具体的放越前面
_CODE_TEXT = (
    ("AuthenticationError", "API Key 无效或已过期，注意火山方舟与 BytePlus 的 Key 不通用"),
    ("InvalidAccountStatus", "账户状态异常"),
    ("AccountOverdue", "账户已欠费，请先充值"),
    ("OperationDenied.ServiceOverdue", "服务已欠费停用，请先充值"),
    ("OperationDenied.ServiceNotOpen", "模型未开通，请先在控制台的「开通管理」中开通该模型"),
    ("ModelNotOpen", "模型未开通，请先在控制台的「开通管理」中开通该模型"),
    ("InvalidEndpointOrModel.ModelIDAccessDisabled", "该账号不能直接使用模型 ID，请在控制台创建推理接入点，并把模型改成接入点 ID（ep-...）"),
    ("InvalidEndpointOrModel", "模型或接入点不存在，请检查模型 ID，火山方舟与 BytePlus 的模型 ID 不同"),
    ("InvalidEndpoint.ClosedEndpoint", "推理接入点已关闭"),
    ("SetLimitExceeded", "已达到账户设置的消费上限，请在控制台调整"),
    ("ModelAccountTpdRateLimitExceeded", "已达到该模型的每日用量上限"),
    ("AccountQuotaExceeded", "账户额度已用尽"),
    ("QuotaExceeded", "额度已用尽，或排队中的任务过多"),
)


def _error_info(body) -> tuple[str, str]:
    """取出错误体中的 (code, message)，兼容 {"error": {code, message}} 与扁平的 {code, message}。"""
    if not isinstance(body, dict):
        return "", response_detail(body)
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    detail = as_str(err.get("message")) or response_detail(body)
    return as_str(err.get("code")), " ".join(detail.split())[:300]


def seedream_error(label: str, status: int, body, headers=None) -> DrawError:
    """方舟的错误响应：区分内容审核、额度用尽（不重试）与限流（可重试）。"""
    base = error_from_response(label, status, body, headers)
    code, detail = _error_info(body)
    if any(marker in code for marker in _MODERATION):
        target = "生成的图片" if code.startswith("Output") else "提示词" if code.startswith("InputText") else "请求内容"
        return DrawError(f"{label} 拒绝生成：{target}未通过内容安全审核，请换个描述再试（{code}）", status=status)
    retryable = base.retryable
    if status == 429:
        # 限流、服务繁忙的请求没有开始生成，可以重试；额度用尽重试也没用
        retryable = not code.startswith(_QUOTA_CODES)
    elif code == "ModelLoading":  # 模型加载中，请求没有开始生成
        retryable = True
    reason = next((text for prefix, text in _CODE_TEXT if code.startswith(prefix)), "")
    if not reason and status == 429 and retryable:
        reason = "请求过于频繁或服务繁忙"
    if not reason and status < 300:
        reason = "没有生成图片"
    if not reason:
        message = str(base)
        if code and code not in message:
            message += f"（{code}）"
    else:
        message = f"{label} {reason}（HTTP {status}{f'，{code}' if code else ''}）"
        if detail:
            message += f"：{detail}"
    return DrawError(message, status=status, retryable=retryable, retry_after=base.retry_after)


# ---------------------------------------------------------------- 后端


class SeedreamBackend(Backend):
    key = "seedream"
    label = "Seedream"
    default_prompt_mode = "natural"
    supports_negative = False
    # 方舟的数据面接口没有可用 API Key 调用的模型列表，模型 ID 写在配置说明里
    supports_models = False
    max_concurrency = None
    doc_url = "https://www.volcengine.com/docs/82379/1541523"

    @property
    def api_key(self) -> str:
        """去掉误粘贴的 Bearer 前缀和空白（含中间的换行），否则请求头非法，httpx 的报错还会带出 Key。"""
        key = as_str(self.section.get("api_key"))
        if key.lower().startswith("bearer "):
            key = key[7:]
        return "".join(key.split())

    @property
    def base_url(self) -> str:
        return normalize_base_url(as_str(self.section.get("base_url")), DEFAULT_BASE_URL)

    @property
    def endpoint(self) -> str:
        """请求地址：只填域名时补 /api/v3，去掉误粘贴的 /images/generations、/chat/completions 等接口路径。"""
        url = self.base_url
        try:
            parts = urlsplit(url)
            host, _ = parts.hostname, parts.port  # 端口不是数字或超出范围时抛出 ValueError
        except ValueError as e:
            raise DrawError(f"Seedream API 地址格式错误：{e}") from e
        if parts.scheme.lower() not in ("http", "https") or not host:
            raise DrawError(f"Seedream API 地址无效，请使用 http:// 或 https:// 开头的地址：{mask_url_credentials(url)}")
        path = parts.path.rstrip("/")
        for suffix in _URL_SUFFIXES:
            if path.lower().endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                break
        return f"{parts.scheme}://{parts.netloc}{path or '/api/v3'}/images/generations"

    @property
    def model(self) -> str:
        return platform_model(as_str(self.section.get("model")) or DEFAULT_MODEL, self.base_url)

    @property
    def family(self) -> str:
        chosen = as_str(self.section.get("model_family")).lower()
        return chosen if chosen in FAMILIES else detect_family(self.model)

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    def size(self, orientation: str) -> str:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        return pick_size(self.section.get(f"{orientation}_size") or DEFAULT_SIZES[orientation], orientation,
                         self.family)

    def missing(self) -> str:
        key = self.api_key
        if not key:
            return "管理员还没有配置 Seedream API Key"
        if not (key.isascii() and key.isprintable()):
            return "Seedream API Key 含有中文、全角或不可见字符，请重新复制"
        try:
            self.endpoint
        except DrawError as e:
            return str(e)
        return ""

    def build_body(self, req: DrawRequest) -> dict:
        rules = FAMILIES[self.family]
        body = {
            "model": self.model,
            # 接口没有负面提示词字段，附加到提示词末尾；也没有 seed / guidance_scale（当前模型均不支持）
            "prompt": final_prompt(req, self.supports_negative),
            "size": self.size(req.orientation),
            "response_format": "b64_json",
            # 接口默认会加「AI 生成」水印，必须显式传 false
            "watermark": as_bool(self.section.get("watermark"), False),
        }
        if rules["output_format"]:
            fmt = as_str(self.section.get("output_format")).lower()
            body["output_format"] = fmt if fmt in ("png", "jpeg") else DEFAULT_OUTPUT_FORMAT
        if rules["fast"] and as_str(self.section.get("optimize_mode")).lower() == "fast":
            body["optimize_prompt_options"] = {"mode": "fast"}
        extra = self.section.get("extra_body")
        body.update(extra if isinstance(extra, dict) else parse_extra_json(as_str(extra), self.label))
        return body

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        try:
            return await self._generate(clients, req)
        except DrawError as e:
            # 中转服务的报错可能原样带上 API Key，展示给用户、写入日志前打码
            key = self.api_key
            if len(key) >= 8 and key in str(e):
                e.args = (str(e).replace(key, mask_secret(key)),)
            raise

    async def _generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        reason = self.missing()
        if reason:
            raise DrawError(reason)
        body = self.build_body(req)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = await with_retries(lambda: send(
            clients.http, "POST", self.endpoint, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=seedream_error, headers=headers, json=body,
        ))
        data = json_body(resp, self.label)
        items = data.get("data") if isinstance(data.get("data"), list) else []
        items = [item for item in items if isinstance(item, dict)]
        item = next((i for i in items if i.get("b64_json") or i.get("url")), None)
        if item is None:
            # 整个请求没有生成图片时，错误放在顶层 error（组图模式下也可能在单张图的 error 里）
            error = next((i["error"] for i in items if i.get("error")), None) or data.get("error")
            if error:
                raise seedream_error(self.label, resp.status_code,
                                     {"error": error} if isinstance(error, dict) else error, resp.headers)
            raise DrawError(f"{self.label} 没有返回图片：{response_detail(data)[:200]}")
        if item.get("b64_json"):
            image = decode_b64_image(item["b64_json"], self.label)
        else:
            # 图片链接 24 小时内有效，是预签名地址，下载时不能带 API Key
            image = await download_image(clients.http, as_str(item["url"]), label=self.label, proxy=self.proxy,
                                         timeout=self.timeout)
        params = {
            "model": body.get("model"),
            "size": as_str(item.get("size")) or as_str(body.get("size")),
            "watermark": body.get("watermark"),
        }
        if body.get("output_format"):
            params["output_format"] = as_str(item.get("output_format")) or body["output_format"]
        if isinstance(body.get("optimize_prompt_options"), dict):
            params["optimize_mode"] = body["optimize_prompt_options"].get("mode")
        request_id = resp.headers.get("x-request-id")
        if request_id:
            params["request_id"] = request_id
        return image_result(image, params)

    def summary(self) -> str:
        return f"{self.label} / {self.model} / 竖图 {self.size('portrait')}"
