"""Seedream（火山方舟 / BytePlus ModelArk）文生图、图片修改与余额查询后端。

绘图：POST {base}/images/generations，Authorization: Bearer <API Key>，一次生成一张，返回 b64_json。
修改图片使用同一个接口，在 JSON 请求体中加 image 字段（data URL，单张为字符串、多张为数组）。
文档（2026-10）：https://www.volcengine.com/docs/82379/1541523

余额：方舟 API Key 不能查询余额，使用费用中心 OpenAPI QueryBalanceAcct（IAM Access Key 签名，
https://www.volcengine.com/docs/6269/1223898 ，签名方法 https://www.volcengine.com/docs/6369/67269 ）。
"""

import base64
import datetime
import hashlib
import hmac
import math
import re
from urllib.parse import quote, urlsplit

from .base import (
    Backend,
    BalanceInfo,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    InputImage,
    as_bool,
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

# ---------------------------------------------------------------- 图片修改的限制

# 各系列修改图片时最多接受的输入图片数；接入点 ID 无法判断系列，按较小的值处理
EDIT_MAX_IMAGES = {"5.0-pro": 10, "5.0-flash": 10, "5.0-lite": 14, "4.5": 14, "4.0": 14, "generic": 10}
MAX_EDIT_IMAGES = max(EDIT_MAX_IMAGES.values())
FAMILY_NAMES = {"5.0-pro": "5.0 pro", "5.0-flash": "5.0 flash", "5.0-lite": "5.0 lite", "4.5": "4.5", "4.0": "4.0",
                "generic": "接入点 / 未识别的模型"}
# 修改图片的尺寸：match 按第一张图片的宽高比计算宽x高（接口没有「跟随原图」的尺寸）；tier 只发送档位，由模型决定
EDIT_SIZE_MODES = ("match", "tier")
DEFAULT_EDIT_SIZE_MODE = "match"
# 输入图片的限制：单张不超过 30 MB（文档没说是 10^6 还是 2^20，取较小值），宽高都大于 14 像素，
# 宽高比 1/16 ~ 16，总像素 196 ~ 6000x6000
INPUT_MAX_BYTES = 30 * 1000 * 1000
INPUT_MIN_PIXELS = 196
INPUT_MAX_PIXELS = 36_000_000
# data URL 中的图片格式必须小写
_DATA_URL_FORMAT = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp", "image/gif": "gif",
                    "image/bmp": "bmp", "image/tiff": "tiff", "image/heic": "heic", "image/heif": "heif"}
# 不能修改图片的模型（只认没有选择模型系列时的模型 ID）
_NO_EDIT_MODELS = (
    (re.compile(r"seededit"), "SeedEdit 3.0 已于 2026-03-31 下线，请改用 doubao-seedream-5-0-pro-260628 等 Seedream 模型"),
    (re.compile(r"seedream-3-0"), "Seedream 3.0 只能文生图且已下线，请改用 Seedream 4.0 及以上的模型"),
)


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


# ---------------------------------------------------------------- 图片修改


def image_data_url(image: InputImage, label: str, index: int) -> str:
    """把输入图片转成 data:image/<小写格式>;base64,... ，格式不支持或超过 30 MB 时报错。"""
    data = image.data or b""
    if not data:  # 空的 data URL 会被接口拒绝（InvalidImageURL.EmptyURL），提前报错
        raise DrawError(f"{label} 第 {index} 张图片没有数据")
    mime = detect_mime(data) or as_str(image.mime).lower()
    fmt = _DATA_URL_FORMAT.get(mime)
    if not fmt:
        raise DrawError(f"{label} 不支持第 {index} 张图片的格式（支持 jpeg / png / webp / bmp / gif 等）")
    if len(data) > INPUT_MAX_BYTES:
        raise DrawError(f"{label} 要求每张图片不超过 30 MB，第 {index} 张过大")
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode('ascii')}"


def check_input_size(image: InputImage, label: str, index: int) -> None:
    """检查输入图片的宽高是否在接口允许的范围内；不知道尺寸时交给接口判断。"""
    width, height = image.width or 0, image.height or 0
    if width <= 0 or height <= 0:
        return
    if width <= 14 or height <= 14 or width * height < INPUT_MIN_PIXELS:
        raise DrawError(f"{label} 要求图片宽高都大于 14 像素，第 {index} 张太小（{width}x{height}）")
    if not 1 / 16 <= width / height <= 16:
        raise DrawError(f"{label} 要求图片宽高比在 1:16 ~ 16:1 之间，第 {index} 张过于细长（{width}x{height}）")
    if width * height > INPUT_MAX_PIXELS:
        raise DrawError(f"{label} 要求图片总像素不超过 6000x6000，第 {index} 张过大（{width}x{height}）")


def png_has_alpha(data: bytes) -> bool:
    """PNG 是否带透明通道：IHDR 的颜色类型为 4 / 6（灰度 + Alpha、RGBA），或在 IDAT 之前有 tRNS 块。"""
    if detect_mime(data) != "image/png":
        return False
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        if kind == b"IHDR":
            if pos + 18 > len(data):
                return False
            if data[pos + 17] in (4, 6):  # 数据区第 10 个字节是颜色类型
                return True
        elif kind == b"tRNS":
            return True
        elif kind in (b"IDAT", b"IEND"):
            return False
        pos += 12 + length
    return False


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


# 输入图片未通过审核的错误码（前缀匹配，越具体的放越前面）
_INPUT_IMAGE_MODERATION = (
    ("InputImageSensitiveContentDetected.PrivacyInformation", "输入的图片可能包含真实人物"),
    ("InputImageSensitiveContentDetected.PolicyViolation", "输入的图片可能涉及版权限制"),
    ("InputImageSensitiveContentDetected", "输入的图片未通过内容安全审核"),
    ("InputImageRiskDetection", "输入的图片未通过内容安全审核"),
)


def seedream_edit_error(label: str, status: int, body, headers=None) -> DrawError:
    """修改图片的错误响应：输入图片审核、图片无法解析、请求体过大（都不重试），其余同 seedream_error。"""
    code, _ = _error_info(body)
    target = next((t for prefix, t in _INPUT_IMAGE_MODERATION if code.startswith(prefix)), "")
    if target:
        return DrawError(f"{label} 拒绝修改：{target}，请换一张图片再试（{code}）", status=status)
    text = ""
    if code.startswith("InvalidImageURL"):
        text = "无法解析输入的图片（Base64 格式不正确、图片数据损坏或格式不支持）"
    elif status == 413:  # 推理接口没有 413 的错误码，网关可能返回 HTML，按状态码判断
        text = "图片数据过大，请减少图片数量或缩小图片"
    if text:
        return DrawError(f"{label} {text}（HTTP {status}{f'，{code}' if code else ''}）", status=status)
    return seedream_error(label, status, body, headers)


# ---------------------------------------------------------------- 余额查询（费用中心 OpenAPI）

BILLING_SERVICE = "billing"
BILLING_VERSION = "2022-01-01"
BILLING_TIMEOUT = 30.0
# 费用中心是全局服务；非地域性产品签名时可以使用任意地域（BytePlus 的地域未经真实密钥验证，可在配置中修改）
BILLING = {
    "volcengine": {"name": "火山引擎", "host": "billing.volcengineapi.com", "region": "cn-beijing", "unit": "元",
                   "console": "https://console.volcengine.com/finance/fund/recharge"},
    # BytePlus 的返回没有币种字段，只有 BytePlus Pte. Ltd. 签约的账户确定是美元，因此不写单位
    "byteplus": {"name": "BytePlus", "host": "billing.byteplusapi.com", "region": "ap-singapore-1", "unit": "",
                 "console": "https://console.byteplus.com/finance/overview"},
}
BILLING_PLATFORMS = ("auto", "volcengine", "byteplus")
# 地域 ID 只有小写字母、数字和连字符（如 cn-beijing、ap-singapore-1）；其他字符会让签名请求头非法
_REGION_RE = re.compile(r"[a-z0-9-]{1,64}")
_CURRENCY_UNIT = {"CNY": "元", "RMB": "元", "USD": "美元"}
# ResponseMetadata.Error.Code 的说明。Error.Message 会原样带出 Access Key，从不展示
_BILLING_CODE_TEXT = {
    "InvalidAccessKey": "Access Key ID 不存在或已停用",
    "InvalidSecretToken": "Access Key 无效",
    "SignatureDoesNotMatch": "签名不匹配，请检查 Secret Access Key 是否正确",
    "InvalidCredential": "Access Key 格式不正确",
    "InvalidAuthorization": "Access Key 格式不正确",
    "InvalidTimestamp": "签名已过期，请检查服务器时间是否准确",
    "AccessDenied": "该 Access Key 所属用户没有查询余额的权限（请授予系统策略 BillingCenterReadOnlyAccess）",
    "FlowLimitExceeded": "查询过于频繁，请稍后再试",
    "ServiceUnavailableTemp": "费用中心暂时不可用，请稍后再试",
    "InternalServiceTimeout": "费用中心响应超时，请稍后再试",
    "InternalServiceError": "费用中心暂时不可用，请稍后再试",
}
# 查询余额是只读请求，限流和网关错误可以安全重试
_BILLING_RETRY_CODES = ("FlowLimitExceeded", "ServiceUnavailableTemp", "InternalServiceTimeout", "InternalServiceError")


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def canonical_query(query: dict) -> str:
    """规范化查询字符串：名称和值按 RFC 3986 编码（空格为 %20，只保留 -_.~ 不编码），按名称排序。"""
    def enc(value) -> str:
        return quote(str(value), safe="-_.~")

    return "&".join(f"{enc(k)}={enc(v)}" for k, v in sorted(query.items(), key=lambda kv: enc(kv[0])))


def canonical_request(method: str, host: str, query: dict, body: bytes, x_date: str) -> str:
    """签名用的规范请求，只签 host 与 x-date 两个请求头（与官方签名文档的示例相同）。"""
    return "\n".join([
        method.upper(), "/", canonical_query(query),
        f"host:{host.strip()}\nx-date:{x_date}\n", "host;x-date", hashlib.sha256(body).hexdigest(),
    ])


def sign_v4(method: str, host: str, query: dict, body: bytes, access_key_id: str, secret_access_key: str,
            region: str, service: str, now: datetime.datetime | None = None) -> dict:
    """火山引擎 / BytePlus OpenAPI 的 HMAC-SHA256 签名，返回需要附加的 X-Date 与 Authorization 请求头。

    https://www.volcengine.com/docs/6369/67269 ：kDate = HMAC(SK, 日期)，依次以地域、服务、"request" 派生签名密钥，
    对 "HMAC-SHA256\\n X-Date \\n 凭证范围 \\n sha256(规范请求)" 签名。
    """
    x_date = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    scope = f"{x_date[:8]}/{region}/{service}/request"
    digest = hashlib.sha256(canonical_request(method, host, query, body, x_date).encode("utf-8")).hexdigest()
    string_to_sign = f"HMAC-SHA256\n{x_date}\n{scope}\n{digest}"
    key = _hmac_sha256(secret_access_key.encode("utf-8"), x_date[:8])
    for part in (region, service, "request"):
        key = _hmac_sha256(key, part)
    signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "X-Date": x_date,
        "Authorization": f"HMAC-SHA256 Credential={access_key_id}/{scope}, SignedHeaders=host;x-date, "
                         f"Signature={signature}",
    }


def billing_error(label: str, status: int, body, headers=None) -> DrawError:
    """费用中心的错误响应 {"ResponseMetadata": {"Error": {"Code", "Message"}}}；只展示错误码对应的说明。"""
    meta = body.get("ResponseMetadata") if isinstance(body, dict) else None
    err = meta.get("Error") if isinstance(meta, dict) and isinstance(meta.get("Error"), dict) else {}
    code = as_str(err.get("Code"))
    reason = _BILLING_CODE_TEXT.get(code) or ("服务器错误" if status >= 500 else "查询失败")
    retry_after = error_from_response(label, status, None, headers).retry_after
    return DrawError(f"{label} 余额查询失败：{reason}（HTTP {status}{f'，{code}' if code else ''}）", status=status,
                     retryable=code in _BILLING_RETRY_CODES or status in (429, 502, 503, 504),
                     retry_after=retry_after)


def _money(value, unit: str) -> str:
    text = as_str(value) or "0"
    try:
        text = f"{float(text):,.2f}"
    except (ValueError, OverflowError):
        pass
    return f"{text} {unit}".strip()


def _nonzero(value) -> bool:
    try:
        return float(as_str(value) or 0) != 0
    except (ValueError, OverflowError):
        return bool(as_str(value))


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
    # 所有模型系列的最大值；实例上按当前模型系列给出（5.0 pro / flash 与接入点 10 张，其余 14 张）
    max_edit_images = MAX_EDIT_IMAGES
    # 没有重绘幅度参数，按修改指令修改图片
    supports_strength = False
    # 需要单独配置 IAM Access Key（方舟 API Key 不能查询余额）
    supports_balance = True
    billing_url = BILLING["volcengine"]["console"]

    def __init__(self, section: dict | None):
        super().__init__(section)
        # 插件按实例上的 max_edit_images 截取输入图片，网页按 billing_url 显示账单页面
        # （family / billing_platform 对任何配置值都不会抛出异常）
        self.max_edit_images = EDIT_MAX_IMAGES.get(self.family, EDIT_MAX_IMAGES["generic"])
        self.billing_url = BILLING[self.billing_platform or "volcengine"]["console"]

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

    # ------------------------------------------------------------ 图片修改

    @property
    def edit_size_mode(self) -> str:
        mode = as_str(self.section.get("edit_size_mode")).lower()
        return mode if mode in EDIT_SIZE_MODES else DEFAULT_EDIT_SIZE_MODE

    def edit_size(self, req: DrawRequest) -> str:
        """修改图片的 size。接口没有「自动 / 跟随原图」的尺寸（auto 只用于图层拆分），只能发档位或宽x高。

        match：按第一张图片的宽高比与该方向配置尺寸的面积计算宽x高，再调整到模型允许的范围；
        画面方向与原图不同（用户明确要求换方向）或不知道原图尺寸时，使用该方向的配置尺寸。
        tier：只发送与配置尺寸面积最接近的档位（如 2K），由模型决定输出尺寸。
        """
        rules = FAMILIES[self.family]
        first = req.images[0] if req.images else None
        shape = image_orientation(first) if first else ""
        orientation = req.orientation if req.orientation in DEFAULT_SIZES else (shape or "portrait")
        configured = self.size(orientation)
        w0, h0 = (int(v) for v in configured.split("x"))
        if self.edit_size_mode == "tier":
            area = w0 * h0
            return min(rules["tiers"], key=lambda t: abs(math.log(area / (TIER_VALUES[t] * 1024) ** 2)))
        if not shape or shape != orientation:
            return configured
        ratio = min(16.0, max(1 / 16, first.width / first.height))
        width = max(16, round(math.sqrt(w0 * h0 * ratio) / 16) * 16)
        height = max(16, round(math.sqrt(w0 * h0 / ratio) / 16) * 16)
        return "%dx%d" % fit_pixels(width, height, *rules["pixels"])

    def edit_missing(self) -> str:
        """所有 Seedream 4.0 及以上的模型都能修改图片；已下线的 SeedEdit 3.0、只能文生图的 3.0 不能。"""
        if self.family == "generic":
            name = re.sub(r"[._\s]", "-", self.model.lower())
            for pattern, reason in _NO_EDIT_MODELS:
                if pattern.search(name):
                    return f"当前模型 {self.model} 不支持修改图片：{reason}"
        return ""

    def build_edit_body(self, req: DrawRequest) -> dict:
        reason = self.edit_missing()
        if reason:
            raise DrawError(reason)
        limit = self.max_edit_images
        if len(req.images) > limit:
            raise DrawError(f"{self.label} {FAMILY_NAMES[self.family]} 修改图片最多使用 {limit} 张图片，"
                            f"本次有 {len(req.images)} 张")
        urls = []
        for index, image in enumerate(req.images, 1):
            check_input_size(image, self.label, index)
            urls.append(image_data_url(image, self.label, index))
        rules = FAMILIES[self.family]
        body = {
            "model": self.model,
            # 修改指令按数组顺序称呼图片（image 1 / 图1）；接口没有负面提示词、重绘幅度、种子、蒙版等字段
            "prompt": final_prompt(req, self.supports_negative),
            "image": urls[0] if len(urls) == 1 else urls,
            "size": self.edit_size(req),
            "response_format": "b64_json",
            "watermark": as_bool(self.section.get("watermark"), False),
        }
        # 与文生图相同：不发送组图开关 sequential_image_generation（5.0 pro / flash 不接受，默认只出一张）
        if rules["output_format"]:
            fmt = as_str(self.section.get("output_format")).lower()
            body["output_format"] = fmt if fmt in ("png", "jpeg") else DEFAULT_OUTPUT_FORMAT
        if rules["fast"] and as_str(self.section.get("optimize_mode")).lower() == "fast":
            body["optimize_prompt_options"] = {"mode": "fast"}
        # 透明背景：仅 5.0 pro / flash、只有一张带透明通道的输入图片时可用，且必须输出 PNG
        if (as_bool(self.section.get("keep_transparency"), False) and self.family in ("5.0-pro", "5.0-flash")
                and len(req.images) == 1 and png_has_alpha(req.images[0].data)):
            body["background"] = "transparent"
            body["output_format"] = "png"
        extra = self.section.get("extra_body")
        body.update(extra if isinstance(extra, dict) else parse_extra_json(as_str(extra), self.label))
        return body

    def plan(self, req: DrawRequest) -> dict:
        if not req.is_edit:
            return super().plan(req)
        params = {"model": self.model, "mode": "edit", "input_images": len(req.images)}
        try:
            params["size"] = self.edit_size(req)
        except DrawError:  # 尺寸无法调整到允许范围时由 generate() 报告
            pass
        return params

    async def _generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        reason = self.missing()
        if reason:
            raise DrawError(reason)
        if req.is_edit:
            body, on_error = self.build_edit_body(req), seedream_edit_error
        else:
            body, on_error = self.build_body(req), seedream_error
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # 只重试连接失败、限流（不含额度用尽）和 502 / 503；超时不重试，服务端可能已经生成并计费
        resp = await with_retries(lambda: send(
            clients.http, "POST", self.endpoint, label=self.label, proxy=self.proxy, timeout=self.timeout,
            on_error=on_error, headers=headers, json=body,
        ))
        data = json_body(resp, self.label)
        items = data.get("data") if isinstance(data.get("data"), list) else []
        items = [item for item in items if isinstance(item, dict)]
        item = next((i for i in items if i.get("b64_json") or i.get("url")), None)
        if item is None:
            # 整个请求没有生成图片时，错误放在顶层 error（组图模式下也可能在单张图的 error 里）
            error = next((i["error"] for i in items if i.get("error")), None) or data.get("error")
            if error:
                raise on_error(self.label, resp.status_code,
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
            # 返回中的 output_format 只有 5.0 pro / flash 提供
            params["output_format"] = as_str(item.get("output_format")) or body["output_format"]
        if isinstance(body.get("optimize_prompt_options"), dict):
            params["optimize_mode"] = body["optimize_prompt_options"].get("mode")
        if req.is_edit:
            params["mode"] = "edit"
            params["input_images"] = len(req.images)
            if body.get("background"):
                params["background"] = body["background"]
        request_id = resp.headers.get("x-request-id")
        if request_id:
            params["request_id"] = request_id
        return image_result(image, params)

    # ------------------------------------------------------------ 余额查询

    @property
    def access_key(self) -> tuple[str, str]:
        """(Access Key ID, Secret Access Key)，去掉误粘贴的空白和换行。"""
        return ("".join(as_str(self.section.get("access_key_id")).split()),
                "".join(as_str(self.section.get("secret_access_key")).split()))

    @property
    def billing_platform(self) -> str:
        """查询哪个平台的费用中心：配置指定，或按 API 地址判断；第三方中转地址返回空串。"""
        chosen = as_str(self.section.get("billing_platform")).lower()
        if chosen in BILLING:
            return chosen
        try:
            host = (urlsplit(self.base_url).hostname or "").lower()
        except ValueError:
            return ""
        if host.endswith("bytepluses.com"):
            return "byteplus"
        if host.endswith("volces.com"):
            return "volcengine"
        return ""

    def balance_missing(self) -> str:
        """方舟 API Key 不能查询余额，因此不检查 missing()，只检查 Access Key 和平台。"""
        platform = self.billing_platform
        if not platform:
            return (f"{self.label} 当前 API 地址不是火山方舟 / BytePlus 的官方地址，无法自动判断余额查询平台；"
                    "使用官方账号时请在配置中选择「余额查询平台」，中转服务的余额请到中转服务的网站查看")
        ak, sk = self.access_key
        if not (ak and sk):
            return (f"{self.label} 的 API Key 不能查询余额：需要在配置中填写 Access Key ID 和 Secret Access Key"
                    f"（建议为只授予 BillingCenterReadOnlyAccess 权限的子用户创建），或到 {BILLING[platform]['console']} 查看")
        if not (ak.isascii() and ak.isprintable() and sk.isascii() and sk.isprintable()):
            return f"{self.label} 的 Access Key 含有中文、全角或不可见字符，请重新复制"
        if not _REGION_RE.fullmatch(self.billing_region):
            return f"{self.label} 的余额查询签名地域格式不正确（应为 cn-beijing、ap-singapore-1 这样的地域 ID），一般留空即可"
        return ""

    @property
    def billing_region(self) -> str:
        """签名用的地域：配置指定（去掉空白、转成小写），留空时使用平台的默认地域。"""
        region = "".join(as_str(self.section.get("billing_region")).split()).lower()
        return region or BILLING[self.billing_platform or "volcengine"]["region"]

    async def balance(self, clients: Clients) -> BalanceInfo:
        try:
            return await self._balance(clients)
        except DrawError as e:
            # 防御：Access Key / Secret 不能出现在展示给用户、写入日志的错误中
            text = str(e)
            for secret in self.access_key:
                if len(secret) >= 8 and secret in text:
                    text = text.replace(secret, mask_secret(secret))
            e.args = (text,)
            raise

    async def _balance(self, clients: Clients) -> BalanceInfo:
        reason = self.balance_missing()
        if reason:
            raise DrawError(reason)
        ak, sk = self.access_key
        platform = self.billing_platform
        cfg = BILLING[platform]
        region = self.billing_region
        query = {"Action": "QueryBalanceAcct", "Version": BILLING_VERSION}
        url = f"https://{cfg['host']}/?{canonical_query(query)}"  # 与签名时相同的编码

        def request():  # 每次重试都重新签名（X-Date 更新，签名默认 15 分钟内有效）
            headers = sign_v4("GET", cfg["host"], query, b"", ak, sk, region, BILLING_SERVICE)
            return send(clients.http, "GET", url, label=self.label, proxy=self.proxy, timeout=BILLING_TIMEOUT,
                        on_error=billing_error, headers=headers)

        resp = await with_retries(request)
        data = json_body(resp, self.label)
        meta = data.get("ResponseMetadata") if isinstance(data.get("ResponseMetadata"), dict) else {}
        if isinstance(meta.get("Error"), dict):
            raise billing_error(self.label, resp.status_code, data, resp.headers)
        result = data.get("Result") if isinstance(data.get("Result"), dict) else {}
        # 没有返回可用余额（或为 null）时报错，不能显示成 0
        if not as_str(result.get("AvailableBalance")):
            raise DrawError(f"{self.label} 余额查询返回的数据格式异常")
        currency = as_str(result.get("Currency")).upper()
        unit = _CURRENCY_UNIT.get(currency, currency or cfg["unit"])
        items = [("可用余额", _money(result.get("AvailableBalance"), unit))]
        if as_str(result.get("CashBalance")):
            items.append(("现金余额", _money(result.get("CashBalance"), unit)))
        for name, key in (("信控额度", "CreditLimit"), ("冻结金额", "FreezeAmount"), ("欠费金额", "ArrearsBalance")):
            if _nonzero(result.get(key)):
                items.append((name, _money(result.get(key), unit)))
        owner = f"{cfg['name']} " if cfg["name"].isascii() else cfg["name"]
        note = (f"{owner}主账号的资金余额（可用余额 = 现金余额 − 冻结金额 + 信控额度 − 欠费金额），所有产品共用，"
                "不是单个 API Key 的额度；代金券、资源包和方舟免费推理额度不在其中")
        if not unit:
            note += "；金额为账户的结算币种（BytePlus Pte. Ltd. 签约的账户为美元）"
        return BalanceInfo(items=items, note=note)

    def summary(self) -> str:
        return f"{self.label} / {self.model} / 竖图 {self.size('portrait')}"
