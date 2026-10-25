"""ComfyUI 后端：本地 / 自部署 ComfyUI 的文生图与图生图（修改图片），也支持 Comfy Cloud（https://cloud.comfy.org）。

本地流程：POST /prompt 提交 API 格式工作流 → 轮询 GET /history/{prompt_id} → GET /view 下载第一张输出图片。
Comfy Cloud：POST /api/prompt（X-API-Key）→ 轮询 GET /api/jobs/{id} → GET /api/view（302 到签名地址，下载时不带 Key）。
修改图片：先 POST /upload/image（Cloud：/api/upload/image，multipart，字段 image / type=input / overwrite）上传输入图片，
把返回的文件名填入 LoadImage 节点，之后与文生图相同。内置工作流为 SDXL 图生图（KSampler.denoise = 重绘幅度）。
余额：Comfy Cloud 的 GET /api/billing/usage/timeseries，或 comfy.org 平台的 GET https://api.comfy.org/customers/balance
（同一个 comfyui- 开头的 Key；本地 ComfyUI 只有配置了付费 API 节点用的 Key 时才有积分余额）。
"""

import asyncio
import copy
import hashlib
import ipaddress
import json
import math
import random
import re
import time
import uuid
from urllib.parse import quote, urlencode, urljoin, urlparse

import httpx

from .base import (
    MAX_RETRY_WAIT,
    Backend,
    BalanceInfo,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    InputImage,
    as_float,
    as_int,
    as_str,
    detect_mime,
    download_image,
    error_from_response,
    image_orientation,
    image_result,
    is_local_url,
    json_body,
    mask_url_credentials,
    normalize_base_url,
    parse_extra_json,
    parse_ratio,
    parse_size,
    response_detail,
    send,
    with_retries,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
CLOUD_HOST = "cloud.comfy.org"
DEFAULT_SIZES = {"portrait": "832x1216", "landscape": "1216x832", "square": "1024x1024"}
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "dpmpp_2m"
DEFAULT_SCHEDULER = "karras"
DEFAULT_TIMEOUT = 300.0
# 配置项 required_positive / required_negative 的默认值（由 main.py 拼接到提示词中），适合 Illustrious / NoobAI 等 SDXL 动漫模型
DEFAULT_REQUIRED_POSITIVE = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
DEFAULT_REQUIRED_NEGATIVE = (
    "nsfw, lowres, worst quality, bad quality, bad anatomy, bad hands, extra digits, fewer digits, "
    "jpeg artifacts, signature, watermark, username, text, logo, blurry"
)
MAX_SEED = 0xFFFFFFFFFFFFFFFF  # KSampler.seed / noise_seed 的上限
MIN_SIDE, MAX_SIDE = 16, 16384  # EmptyLatentImage 宽高范围

# 内置的 SDXL 文生图工作流（API 格式）。整个值是占位符时替换为对应类型的值（数字或文字）
DEFAULT_WORKFLOW = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}", "sampler_name": "{{sampler_name}}",
        "scheduler": "{{scheduler}}", "denoise": 1.0,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "{{model}}"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": "{{width}}", "height": "{{height}}", "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{negative}}", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "astrbot/draw", "images": ["8", 0]}},
}

PLACEHOLDERS = ("prompt", "negative", "negative_prompt", "model", "ckpt_name", "seed", "width", "height",
                "steps", "cfg", "sampler_name", "scheduler")

# ---- 图片修改

# 自定义修改工作流最多使用的图片数：{{image1}}..{{image4}}（{{image}} 同 {{image1}}）；内置图生图工作流只用 1 张
MAX_EDIT_IMAGES = 4
IMAGE_PLACEHOLDERS = ("image",) + tuple(f"image{i}" for i in range(1, MAX_EDIT_IMAGES + 1))
# 图片修改工作流可用的占位符：文生图的全部占位符 + 图片 + 重绘幅度 + 目标总像素（百万像素）
EDIT_PLACEHOLDERS = PLACEHOLDERS + IMAGE_PLACEHOLDERS + ("denoise", "megapixels")
# 重绘幅度：main.py / webapi.py 总会传入 req.strength（LLM 给出的值或 draw.edit_strength），无效时才用这个值
DEFAULT_EDIT_DENOISE = 0.6
UPLOAD_SUBFOLDER = "astrbot"  # 本地 ComfyUI 上传到 input/astrbot/；Comfy Cloud 按内容哈希存储，忽略 subfolder
UPLOAD_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
              "image/bmp": ".bmp"}
UPLOAD_ATTEMPTS = 3
OUTPUT_CLASSES = ("SaveImage", "PreviewImage")
# /object_info 查到的必填输入：(地址, 节点类型) -> (过期时间, 必填输入名；查不到时为 None)
REQUIRED_CACHE_TTL = 600.0
_REQUIRED_CACHE: dict[tuple[str, str], tuple[float, frozenset | None]] = {}
NO_IMAGE_INPUT = "图片修改工作流里没有 {{image}} 占位符，也没有 LoadImage（加载图像）节点"
NO_PROMPT_INPUT = (
    "图片修改工作流里没有 {{prompt}} 占位符，也没能从采样器的 positive 连线找到 CLIPTextEncode 一类的提示词节点。"
    "请把正面提示词改成 \"{{prompt}}\"（CLIPTextEncode 的 text；Qwen-Image-Edit 的 TextEncodeQwenImageEdit(Plus) 是 prompt）"
)

# 内置的 SDXL 图生图工作流（API 格式）：LoadImage → 按总像素缩放（保持原图宽高比）→ VAEEncode → KSampler(denoise<1)。
# 与文生图工作流的节点编号相同，只是把 EmptyLatentImage(5) 换成了 10/11/12
DEFAULT_EDIT_WORKFLOW = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}", "sampler_name": "{{sampler_name}}",
        "scheduler": "{{scheduler}}", "denoise": "{{denoise}}",
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["12", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "{{model}}"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{negative}}", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "astrbot/edit", "images": ["8", 0]}},
    "10": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}},
    # resolution_steps 是较新版本才有的输入：旧版 ComfyUI 忽略未声明的字面量输入，VAEEncode 会把边长裁到 8 的倍数
    "11": {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["10", 0], "upscale_method": "lanczos", "megapixels": "{{megapixels}}", "resolution_steps": 8}},
    "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 2]}},
}

# ---- 余额

COMFY_PLATFORM_API = "https://api.comfy.org"
CREDITS_PER_USD = 211  # 官方换算「211 credits = $1 USD」，不是接口返回值
# 没有单独的账单网页：余额在 Comfy Cloud / ComfyUI 的 设置 → Workspace → Plan & Credits 中，该文档说明了位置
BILLING_URL = "https://docs.comfy.org/interface/credits"

_PH = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
_PH_WHOLE = re.compile(r"^\s*\{\{\s*([A-Za-z_]\w*)\s*\}\}\s*$")
_BILLING = re.compile(r"quota|billing|credit|balance|insufficient|payment|paid|subscription|余额|额度|欠费|充值", re.I)
# Partner / API 节点（GPT Image、Gemini 等）被服务商的内容审核拒绝时的常见说法
_MODERATION = re.compile(r"moderation|content[ _-]?policy|safety (?:system|filter|check)|nsfw|inappropriate|"
                         r"flagged|unsafe content|policy violation|内容审核|违规", re.IGNORECASE)
_CLOUD_DONE = ("completed", "success")
_CLOUD_FAILED = ("failed", "error", "non_retryable_error", "lost", "cancelled")

UI_FORMAT_ERROR = (
    "这是 ComfyUI 的普通工作流（UI 格式，含 nodes / links），提交接口只接受 API 格式。"
    "请在 ComfyUI 中点「工作流 → 导出 (API)」（旧版界面：在设置中开启开发者模式后点 Save (API Format)）导出后再粘贴"
)


# ---------------------------------------------------------------- 工作流


def _quote_bare_placeholders(text: str) -> str:
    """把 JSON 字符串之外的 {{name}}（如 "seed": {{seed}}）加上引号，使其能被解析。"""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                out.append(text[i:i + 2])
                i += 2
                continue
            in_str = ch != '"'
        elif ch == '"':
            in_str = True
        elif text.startswith("{{", i):
            match = _PH.match(text, i)
            if match:
                out.append(f'"{{{{{match.group(1)}}}}}"')
                i = match.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_api_graph(data) -> bool:
    return isinstance(data, dict) and bool(data) and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs", {}), dict)
        for node in data.values()
    )


def load_workflow(raw, *, edit: bool = False) -> dict:
    """解析 API 格式的工作流；留空时返回内置的 SDXL 工作流。

    edit=True 时解析图片修改工作流：留空返回内置的图生图工作流，并允许 {{image}}、{{denoise}} 等占位符。
    """
    default = DEFAULT_EDIT_WORKFLOW if edit else DEFAULT_WORKFLOW
    label = "ComfyUI 图片修改工作流" if edit else "ComfyUI 工作流"
    allowed = EDIT_PLACEHOLDERS if edit else PLACEHOLDERS
    if isinstance(raw, dict):
        data = copy.deepcopy(raw)
    else:
        text = as_str(raw).lstrip("\ufeff")  # 去掉 BOM
        if not text:
            return copy.deepcopy(default)
        try:
            data = json.loads(_quote_bare_placeholders(text))
        except (ValueError, RecursionError) as e:  # 包括 JSONDecodeError、超长数字和嵌套过深
            raise DrawError(f"{label}不是有效的 JSON：{e}") from e
    if not data:
        return copy.deepcopy(default)
    if isinstance(data, dict) and _is_api_graph(data.get("prompt")):
        data = data["prompt"]  # 兼容 {"prompt": {...}} 形式
    if isinstance(data, dict) and ("nodes" in data or "links" in data):
        raise DrawError(UI_FORMAT_ERROR)
    if not _is_api_graph(data):
        bad = [str(k) for k, v in data.items() if not (isinstance(v, dict) and "class_type" in v)] \
            if isinstance(data, dict) else []
        where = f"（异常的键：{', '.join(bad[:5])}）" if bad else ""
        raise DrawError(f"{label}不是 API 格式：每个顶层的值都应是含 class_type 和 inputs 的节点{where}。"
                        "请用「导出 (API)」/ Save (API Format) 导出")
    unknown = sorted(find_placeholders(data) - set(allowed))
    if unknown:
        name = unknown[0]
        message = f"{label}中有未知的占位符 {{{{{name}}}}}，可用：{', '.join(allowed)}"
        if not edit and name in EDIT_PLACEHOLDERS:
            message += "。{{image}}、{{denoise}} 等图片修改用的占位符请写在「图片修改工作流」中"
        elif re.fullmatch(r"image\d+", name):
            message += f"。最多使用 {MAX_EDIT_IMAGES} 张图片（{{{{image1}}}} 到 {{{{image{MAX_EDIT_IMAGES}}}}}）"
        raise DrawError(message)
    return data


def find_placeholders(obj, found: set | None = None) -> set:
    found = set() if found is None else found
    if isinstance(obj, dict):
        for value in obj.values():
            find_placeholders(value, found)
    elif isinstance(obj, list):
        for value in obj:
            find_placeholders(value, found)
    elif isinstance(obj, str) and "{{" in obj:
        found.update(name.lower() for name in _PH.findall(obj))
    return found


def fill_placeholders(obj, values: dict, used: set):
    """在解析后的 JSON 中替换占位符（不直接替换 JSON 文本，提示词中的引号、换行不会破坏结构）。"""
    if isinstance(obj, dict):
        return {k: fill_placeholders(v, values, used) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_placeholders(v, values, used) for v in obj]
    if not isinstance(obj, str) or "{{" not in obj:
        return obj

    def value(name: str):  # 未知的占位符已在 load_workflow 中报错
        name = name.lower()
        used.add(name)
        return values[name]

    whole = _PH_WHOLE.match(obj)
    if whole:  # 整个值就是占位符：替换为数字或文字
        return value(whole.group(1))
    return _PH.sub(lambda m: str(value(m.group(1))), obj)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def apply_fallbacks(graph: dict, values: dict, used: set, model: str) -> str:
    """补全工作流没有使用的占位符，返回实际使用的模型名（无法确定时为空串）：

    - 没有 {{prompt}}：沿采样器的 positive / negative 连线找到 CLIPTextEncode 一类节点填入提示词；
    - 没有 {{seed}}：替换所有固定的 seed / noise_seed，避免 ComfyUI 直接返回缓存的同一张图；
    - 没有 {{width}} / {{height}}：设置 Empty*Latent* 节点的尺寸；
    - 没有 {{model}}：配置了模型且只有一个 ckpt_name 时替换它。
    """
    def text_inputs(ref):
        node = graph.get(str(ref[0])) if isinstance(ref, list) and len(ref) == 2 else None
        inputs = node.get("inputs") if isinstance(node, dict) else None
        return inputs if isinstance(inputs, dict) and isinstance(inputs.get("text"), str) else None

    need_prompt = "prompt" not in used
    need_negative = need_prompt and not used & {"negative", "negative_prompt"} and bool(values["negative"])
    wired = False
    ckpt_inputs = []
    for node in graph.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if need_prompt and "positive" in inputs:
            target = text_inputs(inputs["positive"])
            if target is not None:
                target["text"] = values["prompt"]
                wired = True
            target = text_inputs(inputs.get("negative")) if need_negative else None
            if target is not None:
                target["text"] = values["negative"]
        if "seed" not in used:
            for key in ("seed", "noise_seed"):
                if _is_int(inputs.get(key)):
                    inputs[key] = values["seed"]
        class_type = str(node.get("class_type"))
        if not used & {"width", "height"} and class_type.startswith("Empty") and "Latent" in class_type:
            for key in ("width", "height"):
                if _is_int(inputs.get(key)):
                    inputs[key] = values[key]
        if isinstance(inputs.get("ckpt_name"), str):
            ckpt_inputs.append(inputs)
    if need_prompt and not wired:
        raise DrawError("ComfyUI 工作流里没有 {{prompt}} 占位符，也没能从采样器的 positive 连线找到提示词节点。"
                        "请把正面提示词节点（CLIPTextEncode）的 text 改成 \"{{prompt}}\"")
    if used & {"model", "ckpt_name"}:
        return model
    if len(ckpt_inputs) == 1:
        if model:
            ckpt_inputs[0]["ckpt_name"] = model
        return ckpt_inputs[0]["ckpt_name"]
    return ""  # 配置的模型没有用到（如 Flux 的 UNETLoader、多个 Checkpoint 加载节点），不记录


def fit_size(text: str, default: str) -> tuple[int, int]:
    """把配置的尺寸换算成 ComfyUI 接受的宽高：支持 832x1216，或 3:4 这样的比例（按约 100 万像素换算），
    对齐到 8 的倍数并限制在 16-16384。"""
    size = parse_size(text, (0, 0))
    if size == (0, 0):
        ratio = parse_ratio(text)
        if ratio:
            scale = math.sqrt(1024 * 1024 / (ratio[0] * ratio[1]))
            size = (round(ratio[0] * scale / 64) * 64, round(ratio[1] * scale / 64) * 64)
        else:
            size = parse_size(default)
    width, height = (max(MIN_SIDE, min(MAX_SIDE, round(v / 8) * 8)) for v in size)
    return width, height


def pick_image(outputs: dict) -> dict:
    """取第一张保存的图片（SaveImage，type=output），没有时退而取预览图（type=temp）。"""
    images = [
        image
        for output in (outputs.values() if isinstance(outputs, dict) else [])
        if isinstance(output, dict) and isinstance(output.get("images"), list)
        for image in output["images"]
        if isinstance(image, dict) and as_str(image.get("filename"))
    ]
    if not images:
        raise DrawError("ComfyUI 任务已完成，但没有图片输出：工作流中需要有 SaveImage（保存图像）节点")
    return sorted(images, key=lambda image: image.get("type") != "output")[0]


def format_prompt_error(body: dict) -> str:
    """整理提交被拒绝时的说明：{"error": {type, message, details}, "node_errors": {id: {errors, class_type}}}。"""
    parts = []
    err = body.get("error")
    if isinstance(err, dict):
        text = as_str(err.get("message")) or as_str(err.get("type"))
        details = as_str(err.get("details"))
        if details and err.get("type") != "prompt_outputs_failed_validation":
            text += f"（{details}）"
        if err.get("type") == "missing_node_type":
            text += "。请在 ComfyUI 中安装工作流用到的自定义节点"
        parts.append(text)
    elif err:
        parts.append(as_str(err))
    node_errors = body.get("node_errors")
    for node_id, info in node_errors.items() if isinstance(node_errors, dict) else []:
        errors = info.get("errors") if isinstance(info, dict) else None
        for item in errors if isinstance(errors, list) else []:
            if not isinstance(item, dict):
                continue
            extra = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
            field = f".{extra['input_name']}" if extra.get("input_name") else ""
            details = as_str(item.get("details"))
            parts.append(f"节点 #{node_id} {as_str(info.get('class_type'))}{field}：{as_str(item.get('message'))}"
                         + (f"（{details}）" if details else ""))
    if not parts:
        parts.append(response_detail(body))
    return " ".join("；".join(p for p in parts if p).split())[:600]


def format_exec_error(data: dict, label: str = "ComfyUI") -> str:
    """整理执行失败的说明（history 的 execution_error 或 Comfy Cloud 的 execution_error）。"""
    message = " ".join(as_str(data.get("exception_message")).split())[:300]
    text = (f"{label} 执行失败：节点 #{as_str(data.get('node_id'))}（{as_str(data.get('node_type'))}）"
            f"{as_str(data.get('exception_type'))}：{message}")
    if "out of memory" in message.lower():
        text += "。显存不足，可以减小尺寸后重试"
    elif _MODERATION.search(message):
        text += "。内容可能被安全审核拦截了，可以换个说法或换张图片再试"
    return text


def format_history_error(status: dict) -> str:
    for item in status.get("messages") or []:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict)):
            continue
        event, data = item
        if event == "execution_error":
            return format_exec_error(data)
        if event == "execution_interrupted":
            return f"ComfyUI 任务被中断（节点 #{as_str(data.get('node_id'))} {as_str(data.get('node_type'))}）"
    return "ComfyUI 执行失败（没有返回错误详情）"


def _host(url: str) -> str:
    """地址中的主机名（小写）；地址无法使用（如 http://[bad、端口不是数字）时返回空串。"""
    try:
        parts = urlparse(url)
        parts.port  # 端口不是数字或超出范围时抛出 ValueError
        httpx.URL(url)  # 按实际发送请求时的规则再检查一次（如主机名含全角空格）
    except (ValueError, httpx.InvalidURL):
        return ""
    return (parts.hostname or "").lower()


def _plain_http(host: str) -> bool:
    """未写协议时是否按 http 处理：本机、局域网、IP 地址和 Docker 服务名（不含点）通常没有 https。"""
    if "." not in host or is_local_url(f"http://{host}"):
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


class _Redirect(DrawError):
    """Comfy Cloud /api/view 的 302 跳转，location 为签名下载地址。"""

    def __init__(self, location: str):
        super().__init__("重定向")
        self.location = location


# ---------------------------------------------------------------- 图片修改工作流处理


def _is_link(value) -> bool:
    """API 格式中节点之间的连线：["节点编号", 输出序号]。"""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) \
        and _is_int(value[1])


def image_slot_count(graph: dict) -> int:
    """工作流中用到的最大图片序号（{{image}} 与 {{image1}} 都是第 1 张），0 表示没有图片占位符。"""
    used = [1 if name == "image" else int(name[5:]) for name in find_placeholders(graph) if name in IMAGE_PLACEHOLDERS]
    return max(used, default=0)


def load_image_nodes(graph: dict) -> list[str]:
    """没有图片占位符时按节点编号顺序填入图片的 LoadImage 节点。"""
    ids = [nid for nid, node in graph.items()
           if node.get("class_type") == "LoadImage" and isinstance(node.get("inputs", {}).get("image"), str)]
    return sorted(ids, key=lambda nid: (not str(nid).isdigit(), int(nid) if str(nid).isdigit() else 0, str(nid)))


def nodes_using(graph: dict, names: set) -> set:
    """有输入值整体就是这些占位符之一的节点（如 image 为 "{{image3}}" 的 LoadImage）。"""
    found = set()
    for nid, node in graph.items():
        for value in node.get("inputs", {}).values():
            match = _PH_WHOLE.match(value) if isinstance(value, str) else None
            if match and match.group(1).lower() in names:
                found.add(nid)
    return found


def downstream_classes(graph: dict, dead: set) -> set[str]:
    """dead 节点下游（直接或间接用到它们输出）的节点类型，只为这些类型查询 /object_info。"""
    seen, queue, classes = set(), [str(nid) for nid in dead], set()
    while queue:
        nid = queue.pop()
        for oid, node in graph.items():
            if oid not in seen and any(_is_link(v) and str(v[0]) == nid for v in node.get("inputs", {}).values()):
                seen.add(oid)
                queue.append(oid)
                classes.add(str(node.get("class_type")))
    return classes


def required_from_info(info) -> frozenset | None:
    """/object_info 中一个节点类型的必填输入名；格式不对时返回 None。"""
    inputs = info.get("input") if isinstance(info, dict) else None
    required = inputs.get("required") if isinstance(inputs, dict) else None
    return frozenset(required) if isinstance(required, dict) else None


def prune_nodes(graph: dict, dead: set, required: dict | None = None) -> set:
    """删除没有对应图片的加载节点（用户发来的图片比工作流的图片槽少时），返回删掉的节点编号。

    引用被删节点的连线一并去掉。下游节点是否一起删：
    - 知道该节点类型的必填输入（required[class_type]，来自 /object_info）时：丢了必填输入就删
      （如 VAEEncode.pixels），只丢了可选输入就保留（如 TextEncodeQwenImageEditPlus.image2、ReferenceLatent.latent）；
    - 不知道时：去掉后不再有任何连线输入就删。
    """
    removed = set()
    queue = [str(nid) for nid in dead]
    while queue:
        nid = queue.pop()
        if graph.pop(nid, None) is None:
            continue
        removed.add(nid)
        for oid, node in list(graph.items()):
            inputs = node.get("inputs", {})
            hit = [key for key, value in inputs.items() if _is_link(value) and str(value[0]) == nid]
            if not hit:
                continue
            for key in hit:
                del inputs[key]
            need = (required or {}).get(str(node.get("class_type")))
            if need is not None:
                if any(key in need for key in hit):
                    queue.append(oid)
            elif not any(_is_link(v) for v in inputs.values()):
                queue.append(oid)
    return removed


def megapixels_of(width: int, height: int) -> float:
    """宽高对应的百万像素数（ImageScaleToTotalPixels 的 megapixels，范围 0.01-16）。"""
    return round(max(0.01, min(16.0, width * height / (1024 * 1024))), 2)


def fit_to_image(image: InputImage, megapixels: float, step: int) -> tuple[int, int]:
    """按输入图片的宽高比换算出约 megapixels 百万像素的尺寸（对齐到 step）；不知道图片尺寸时返回 (0, 0)。

    step=8 时与 ImageScaleToTotalPixels(resolution_steps=8) 的计算方式一致。
    """
    if not (image.width and image.height):
        return 0, 0
    scale = math.sqrt(megapixels * 1024 * 1024 / (image.width * image.height))
    return tuple(max(step, min(MAX_SIDE, round(side * scale / step) * step)) for side in (image.width, image.height))


def add_alpha_flatten(graph: dict, loader: str, image: InputImage) -> bool:
    """把透明图片铺到白底上再图生图（只用于内置工作流），返回是否插入了节点。

    LoadImage 只把 alpha 输出为 MASK（= 1 - alpha），IMAGE 中透明区域是像素里残留的颜色（常见为黑色），
    图生图会把它当成黑色背景。backends/ 不能用 Pillow，所以在 ComfyUI 里合成：
    ImageCompositeMasked(destination=EmptyImage(白色), source=LoadImage.IMAGE, mask=InvertMask(LoadImage.MASK))。
    插件只把有透明通道的图片保存为 PNG，因此只在 PNG 且知道宽高时插入。
    """
    if (detect_mime(image.data) or image.mime) != "image/png" or not (image.width and image.height):
        return False
    consumers = [(node, key) for node in graph.values() for key, value in node.get("inputs", {}).items()
                 if value == [loader, 0]]
    white, invert, comp = [str(n) for n in range(100, 104 + len(graph)) if str(n) not in graph][:3]
    graph[white] = {"class_type": "EmptyImage", "inputs": {
        "width": image.width, "height": image.height, "batch_size": 1, "color": 0xFFFFFF}}
    graph[invert] = {"class_type": "InvertMask", "inputs": {"mask": [loader, 1]}}
    graph[comp] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": [white, 0], "source": [loader, 0], "x": 0, "y": 0, "resize_source": True,
        "mask": [invert, 0]}}
    for node, key in consumers:
        node["inputs"][key] = [comp, 0]
    return True


# ---------------------------------------------------------------- 余额


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _money(cents: float, currency: str) -> str:
    amount = cents / 100  # 两个接口的余额字段单位都是「分」，即使字段名叫 *_micros
    if currency.upper() in ("", "USD"):
        # 官方说明：付费 API 节点事后扣费，余额可能暂时为负
        return f"{'-' if amount < 0 else ''}${abs(amount):,.2f}"
    return f"{amount:,.2f} {currency.upper()}"


def balance_items(balance: dict, *, spend_usd: float | None = None) -> list[tuple[str, str]]:
    """整理余额字段：优先读 *_cents，其次 *_micros（api.comfy.org 已注明两者相同、单位都是分）。"""
    currency = as_str(balance.get("currency")) or "USD"

    def cents(name: str) -> float | None:
        value = _number(balance.get(f"{name}_cents"))
        return value if value is not None else _number(balance.get(f"{name}_micros"))

    total = cents("amount")
    if total is None:
        raise DrawError("Comfy 返回的余额数据中没有 amount_cents / amount_micros")
    text = _money(total, currency)
    if currency.upper() in ("", "USD"):
        text += f"（约 {round(total / 100 * CREDITS_PER_USD):,} 积分）"
    items = [("剩余额度", text)]
    for key, label in (("cloud_credit_balance", "订阅赠送额度"), ("prepaid_balance", "充值额度"),
                       ("pending_charges", "待结算费用"), ("effective_balance", "扣除待结算后")):
        value = cents(key)
        if value is not None:
            items.append((label, _money(value, currency)))
    if spend_usd is not None:
        items.append(("近一个月消费", _money(spend_usd * 100, "USD")))
    return items


class _PerConfig:
    """按当前配置计算的能力值。

    从类上读取时是该平台的最大能力（与其他后端的类属性一致，如 max_edit_images = 4）；
    从实例上读取时按配置计算（如内置图生图工作流只用 1 张图）。main.py / webapi.py 都通过实例读取。
    """

    def __init__(self, default, compute):
        self.default = default
        self.compute = compute

    def __get__(self, obj, owner=None):
        if obj is None:
            return self.default
        try:
            return self.compute(obj)
        except Exception:  # noqa: BLE001 - 设置页面读取能力时不能因为配置有误而出错
            return self.default


# ---------------------------------------------------------------- 后端


class ComfyUIBackend(Backend):
    key = "comfyui"
    label = "ComfyUI"
    # 内置工作流面向 SDXL 动漫模型（Illustrious / NoobAI 等），使用 Danbooru 标签
    default_prompt_mode = "tags"
    supports_negative = True
    supports_models = True
    # 图片修改：内置图生图工作流 1 张，自定义修改工作流最多 4 张（按工作流中的 {{image1}}..{{image4}} 计算）
    max_edit_images = _PerConfig(MAX_EDIT_IMAGES, lambda self: self.edit_capacity())
    # 内置工作流用 {{denoise}}（重绘幅度）；自定义修改工作流用到 {{denoise}} 时才需要重绘幅度
    supports_strength = _PerConfig(True, lambda self: self.edit_uses_denoise())
    # Comfy Cloud 与配置了 comfy.org Key 的本地 ComfyUI 可以查询积分余额；其他情况 balance_missing() 说明原因
    supports_balance = True
    doc_url = "https://docs.comfy.org/development/comfyui-server/comms_routes"
    billing_url = BILLING_URL
    # 查询任务状态的间隔（秒）
    poll_interval = 1.0
    cloud_poll_interval = 2.0

    @property
    def base_url(self) -> str:
        # 去掉从浏览器地址栏复制时带上的 ?参数 和 #锚点
        url = as_str(self.section.get("base_url")).split("#", 1)[0].split("?", 1)[0].rstrip("/")
        if url and "://" not in url and _plain_http(_host(f"http://{url}")):
            url = f"http://{url}"  # 其余地址交给 normalize_base_url 补 https://
        url = normalize_base_url(url, DEFAULT_BASE_URL)
        if _host(url) == CLOUD_HOST:
            # Comfy Cloud 只提供 https；/api 前缀由各请求自行添加
            url = "https://" + url.split("://", 1)[1].removesuffix("/api")
        return url

    @property
    def cloud(self) -> bool:
        return _host(self.base_url) == CLOUD_HOST

    @property
    def name(self) -> str:
        return "Comfy Cloud" if self.cloud else "ComfyUI"

    @property
    def api_key(self) -> str:
        return as_str(self.section.get("api_key"))

    @property
    def headers(self) -> dict:
        key = self.api_key
        if not key:
            return {}
        if self.cloud:
            return {"X-API-Key": key}
        # ComfyUI 本身没有鉴权，Key 用于前面的反向代理；填写了完整的 "Basic xxx" 等值时原样发送
        return {"Authorization": key if key.lower().startswith(("bearer ", "basic ")) else f"Bearer {key}"}

    @property
    def timeout(self) -> float:
        return as_float(self.section.get("timeout"), DEFAULT_TIMEOUT, 10.0, 1800.0)

    @property
    def steps(self) -> int:
        return as_int(self.section.get("steps"), DEFAULT_STEPS, 1, 10000)

    @property
    def cfg(self) -> float:
        return as_float(self.section.get("cfg"), DEFAULT_CFG, 0.0, 100.0)

    @property
    def sampler(self) -> str:
        return as_str(self.section.get("sampler_name")) or DEFAULT_SAMPLER

    @property
    def scheduler(self) -> str:
        return as_str(self.section.get("scheduler")) or DEFAULT_SCHEDULER

    @property
    def custom_workflow(self) -> bool:
        raw = self.section.get("workflow_json")
        return bool(raw) if isinstance(raw, dict) else bool(as_str(raw))

    def size(self, orientation: str) -> tuple[int, int]:
        orientation = orientation if orientation in DEFAULT_SIZES else "portrait"
        default = DEFAULT_SIZES[orientation]
        return fit_size(as_str(self.section.get(f"{orientation}_size")) or default, default)

    def extra_body(self) -> dict:
        raw = self.section.get("extra_body")
        return dict(raw) if isinstance(raw, dict) else parse_extra_json(as_str(raw), self.label)

    def missing(self) -> str:
        if not _host(self.base_url):
            return f"ComfyUI 地址格式不正确：{mask_url_credentials(self.base_url)}"
        if self.cloud and not self.api_key:
            return "管理员还没有配置 Comfy Cloud API Key"
        try:
            load_workflow(self.section.get("workflow_json"))
        except DrawError as e:
            return str(e)
        return ""

    def summary(self) -> str:
        width, height = self.size("portrait")
        flow = "自定义工作流" if self.custom_workflow else f"{self.steps} 步"
        return f"{self.name} / {self.model or '自动选择模型'} / 竖图 {width}x{height} / {flow}"

    # ------------------------------------------------------------ 图片修改：配置

    @property
    def custom_edit_workflow(self) -> bool:
        raw = self.section.get("edit_workflow_json")
        return bool(raw) if isinstance(raw, dict) else bool(as_str(raw))

    def edit_template(self) -> dict:
        """图片修改工作流：配置了 edit_workflow_json 时使用它，否则使用内置的 SDXL 图生图工作流。"""
        return load_workflow(self.section.get("edit_workflow_json"), edit=True)

    def edit_capacity(self) -> int:
        """当前修改工作流最多使用的图片数；工作流有误时为 1（edit_missing() 会说明原因）。"""
        try:
            graph = self.edit_template()
        except DrawError:
            return 1
        return max(1, min(MAX_EDIT_IMAGES, image_slot_count(graph) or len(load_image_nodes(graph))))

    def edit_uses_denoise(self) -> bool:
        try:
            return "denoise" in find_placeholders(self.edit_template())
        except DrawError:
            return True

    def edit_ready_missing(self) -> str:
        """修改图片前的基本检查：只检查地址和 Cloud Key，文生图工作流有误不影响修改图片。"""
        if not _host(self.base_url):
            return f"ComfyUI 地址格式不正确：{mask_url_credentials(self.base_url)}"
        if self.cloud and not self.api_key:
            return "管理员还没有配置 Comfy Cloud API Key"
        return ""

    def edit_missing(self) -> str:
        # 不调用 missing()：它还会检查文生图工作流，文生图工作流有误不应影响修改图片
        if not _host(self.base_url):
            return f"ComfyUI 地址格式不正确：{mask_url_credentials(self.base_url)}"
        if self.cloud and not self.api_key:
            return "管理员还没有配置 Comfy Cloud API Key"
        try:
            self.extra_body()
            graph = self.edit_template()
        except DrawError as e:
            return f"图片修改配置有误：{e}"
        if not image_slot_count(graph) and not load_image_nodes(graph):
            return NO_IMAGE_INPUT
        # 按生成时的方式试填一遍（只改副本）：提示词既没有占位符也连不到时，生成必然失败，提前说明，不上传图片
        values = self._values(DrawRequest("x", "x"), "", 1, 1024, 1024)
        values.update({name: "x.png" for name in IMAGE_PLACEHOLDERS}, denoise=DEFAULT_EDIT_DENOISE, megapixels=1.0)
        used: set = set()
        try:
            apply_fallbacks(fill_placeholders(graph, values, used), values, used, "")
        except DrawError:
            return NO_PROMPT_INPUT
        return ""

    def edit_megapixels(self, orientation: str) -> float:
        """修改后的总像素与文生图该方向的尺寸设置一致（默认竖图 832x1216 ≈ 0.96 百万像素）。"""
        return megapixels_of(*self.size(orientation))

    # ------------------------------------------------------------ HTTP

    def map_error(self, label: str, status: int, body, headers) -> DrawError:
        if status in (301, 302, 303, 307, 308) and headers.get("location"):
            return _Redirect(headers["location"])
        if status == 400 and isinstance(body, dict) and ("node_errors" in body or isinstance(body.get("error"), dict)):
            return DrawError(f"{label} 拒绝了工作流（HTTP 400）：{format_prompt_error(body)}", status=status)
        detail = response_detail(body)
        if self.cloud and status in (402, 429):
            # Comfy Cloud：402 = 余额不足，429 = 订阅未付费（不是限流），重试也不会成功
            reason = "账户余额不足" if status == 402 else "订阅未激活或未付费"
            return DrawError(f"{label} {reason}（HTTP {status}）" + (f"：{detail}" if detail else ""), status=status)
        err = error_from_response(label, status, body, headers)
        if status == 429 and _BILLING.search(detail):
            err.retryable = False  # 额度 / 余额耗尽，不是暂时的限流
        return err

    def url(self, path: str) -> str:
        base = self.base_url
        if not _host(base):
            raise DrawError(f"ComfyUI 地址格式不正确：{mask_url_credentials(base)}")
        return f"{base}{path}"

    # 不用 get / post 作方法名：基类的 get(name) 用于读取配置
    async def api_get(self, http, path: str, *, attempts: int = 3, timeout: float = 30.0):
        """GET 不会重复生成图片，网络错误和 5xx 都可以重试。"""
        url = self.url(path)
        for attempt in range(1, attempts + 1):
            try:
                return await send(http, "GET", url, label=self.name, proxy=self.proxy,
                                  timeout=timeout, headers=self.headers, on_error=self.map_error)
            except DrawError as e:
                if attempt >= attempts or not (e.retryable or e.status is None or e.status >= 500):
                    raise
                await asyncio.sleep(attempt)
        raise AssertionError("unreachable")

    async def api_post(self, http, path: str, payload=None, *, timeout: float = 10.0):
        return await send(http, "POST", self.url(path), label=self.name, proxy=self.proxy,
                          timeout=timeout, headers=self.headers, json=payload, on_error=self.map_error)

    async def fetch_models(self, http) -> list[str]:
        path = "/api/experiment/models/checkpoints" if self.cloud else "/models/checkpoints"
        try:
            resp = await self.api_get(http, path)
        except DrawError as e:
            if e.status == 404:
                raise DrawError(f"{self.name} 不支持查询模型列表（{path} 返回 404），请在配置中填写模型文件名",
                                status=404) from e
            raise
        try:
            data = resp.json()
        except ValueError as e:
            raise DrawError(f"{self.name} 返回的模型列表不是 JSON") from e
        if not isinstance(data, list):
            raise DrawError(f"{self.name} 返回的模型列表格式异常")
        # 本地返回文件名列表；Comfy Cloud 返回 [{"name": ..., "pathIndex": ...}]
        names = [item.get("name") if isinstance(item, dict) else item for item in data]
        return [name for name in names if isinstance(name, str) and name]

    async def list_models(self, clients: Clients) -> list[str]:
        return await self.fetch_models(clients.http)

    async def default_model(self, http) -> str:
        """模型留空时使用 ComfyUI 模型列表中的第一个。"""
        models = await self.fetch_models(http)
        if not models:
            raise DrawError(f"{self.name} 的 checkpoints 目录中没有模型，请先放入 SDXL 模型，或在配置中填写模型文件名")
        return models[0]

    # ------------------------------------------------------------ 生成

    async def generate(self, clients: Clients, req: DrawRequest) -> DrawResult:
        try:
            if req.is_edit:
                return await self._edit(clients.http, req)
            return await self._generate(clients.http, req)
        except DrawError as e:
            if not (e.retryable and e.status is None and not self.cloud):
                raise
            # 连接失败：ComfyUI 部署方式各异，补充最常见的原因
            raise DrawError(
                f"{e}。请确认 ComfyUI 已启动且地址 {mask_url_credentials(self.base_url)} 正确"
                "（AstrBot 在 Docker 中运行时 127.0.0.1 指容器自身，需要用宿主机 / 局域网 IP，并用 --listen 启动 ComfyUI）",
                retryable=e.retryable,
            ) from e

    @staticmethod
    def _seed(req: DrawRequest) -> int:
        return req.seed if isinstance(req.seed, int) and 0 < req.seed <= MAX_SEED else random.randint(1, 2**32 - 1)

    async def _model(self, http, graph: dict) -> str:
        """配置的模型；留空且工作流用到 {{model}} 时使用 ComfyUI 模型列表中的第一个。"""
        if not self.model and find_placeholders(graph) & {"model", "ckpt_name"}:
            return await self.default_model(http)
        return self.model

    def _values(self, req: DrawRequest, model: str, seed: int, width: int, height: int) -> dict:
        return {
            "prompt": req.prompt,
            "negative": req.negative or "",
            "negative_prompt": req.negative or "",
            "model": model,
            "ckpt_name": model,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": self.steps,
            "cfg": self.cfg,
            "sampler_name": self.sampler,
            "scheduler": self.scheduler,
        }

    @staticmethod
    def _sampler_params(values: dict, used: set) -> dict:
        names = {"steps": "steps", "cfg": "cfg", "sampler_name": "sampler", "scheduler": "scheduler"}
        return {names[k]: values[k] for k in names if k in used}

    async def _generate(self, http, req: DrawRequest) -> DrawResult:
        graph = load_workflow(self.section.get("workflow_json"))
        extra = self.extra_body()  # 先解析，配置有误时不发出请求
        width, height = self.size(req.orientation)
        seed = self._seed(req)
        model = await self._model(http, graph)
        values = self._values(req, model, seed, width, height)
        used: set = set()
        graph = fill_placeholders(graph, values, used)
        model = apply_fallbacks(graph, values, used, model)

        image, pid, warning = await self.run_graph(http, graph, extra)
        params = {
            "model": model,
            "size": f"{width}x{height}",
            "seed": seed,
            "workflow": "custom" if self.custom_workflow else "builtin",
            "prompt_id": pid,
        }
        params.update(self._sampler_params(values, used))
        return image_result(image, params, warning)

    async def run_graph(self, http, graph: dict, extra: dict) -> tuple[bytes, str, str]:
        """提交工作流 → 等待完成 → 下载第一张输出图片，返回 (图片, prompt_id, 警告)。文生图与修改图片共用。"""
        body = {"prompt": graph}
        if not self.cloud:
            # 客户端生成 prompt_id（新版要求标准 UUID）：提交超时或取消时可以按它查找、取消任务
            body.update(client_id=uuid.uuid4().hex, prompt_id=str(uuid.uuid4()))
        body.update(extra)
        extra_data = body.get("extra_data", {})
        if self.cloud and self.api_key and isinstance(extra_data, dict):
            # 官方文档：直接调用 API 提交含付费 API 节点（Partner Nodes）的工作流时须在 extra_data.api_key_comfy_org
            # 中带上 comfy.org Key，它与 X-API-Key 是同一个 Key（浏览器前端会自动带上）。额外请求参数中填写的值优先；
            # 新建 dict，不修改配置中的对象
            body["extra_data"] = {"api_key_comfy_org": self.api_key, **extra_data}
        pid = as_str(body.get("prompt_id"))
        deadline = time.monotonic() + self.timeout
        try:
            pid, warning = await self.submit(http, body, pid)
            outputs = await (self.wait_cloud if self.cloud else self.wait_local)(http, pid, deadline)
        except asyncio.CancelledError:
            # 插件停用或外层超时：同时取消服务端任务，避免显卡继续空跑（Cloud 继续计费）
            if pid:
                await self.cancel_quietly(http, pid)
            raise
        return await self.fetch_image(http, pick_image(outputs)), pid, warning

    # ------------------------------------------------------------ 图片修改

    def _edit_size(self, req: DrawRequest, first: InputImage, custom: bool) -> tuple[int, int, str, bool]:
        """修改图片的尺寸，返回 (宽, 高, 记录用的尺寸, 是否需要换方向)。

        默认保持第 1 张图的宽高比、总像素与该方向的尺寸设置相同；明确要求了与原图不同的方向时
        （req.orientation 与原图方向不同）改用该方向的尺寸。
        """
        megapixels = self.edit_megapixels(req.orientation)
        own = image_orientation(first)
        reshape = bool(own) and req.orientation in DEFAULT_SIZES and req.orientation != own
        if own and not reshape:
            width, height = fit_to_image(first, megapixels, 64)
        else:
            width, height = self.size(req.orientation)
        size = f"{width}x{height}"
        if not custom and not reshape:
            # 内置工作流按总像素缩放：与 ImageScaleToTotalPixels 一致；不知道原图尺寸时只能记录总像素
            size = "x".join(map(str, fit_to_image(first, megapixels, 8))) if own else f"{megapixels}MP"
        return width, height, size, reshape

    def plan(self, req: DrawRequest) -> dict:
        if not req.is_edit:
            return super().plan(req)
        params = {"mode": "edit", "model": self.model}
        try:
            graph = self.edit_template()
            params["input_images"] = min(len(req.images), self.max_edit_images)
            params["size"] = self._edit_size(req, req.images[0], graph != DEFAULT_EDIT_WORKFLOW)[2]
        except Exception:
            pass  # 配置有误时由 generate() 报告
        return params

    async def _edit(self, http, req: DrawRequest) -> DrawResult:
        """上传输入图片 → 填入修改工作流的 LoadImage → 与文生图相同地提交、等待、取图。"""
        reason = self.edit_missing()  # 配置有误时不发出任何请求（包括上传）
        if reason:
            raise DrawError(reason)
        graph = self.edit_template()
        extra = self.extra_body()  # 先解析，配置有误时不发出请求
        # 按实际使用的工作流判断：填了 "{}" 等空工作流时用的仍是内置工作流（需要白底合成、换方向裁切）
        custom = graph != DEFAULT_EDIT_WORKFLOW
        slots = image_slot_count(graph)
        loaders = [] if slots else load_image_nodes(graph)[:MAX_EDIT_IMAGES]
        capacity = min(MAX_EDIT_IMAGES, slots or len(loaders))
        if not capacity:
            raise DrawError(NO_IMAGE_INPUT)
        images = list(req.images)[:capacity]
        ignored = len(req.images) - len(images)
        # 图片比工作流的图片槽少：删掉多余的加载节点和只为它们服务的节点（留空会让 ComfyUI 校验失败）
        if slots:
            dead = nodes_using(graph, {f"image{i}" for i in range(len(images) + 1, MAX_EDIT_IMAGES + 1)})
        else:
            dead = set(loaders[len(images):])
        await self.prune_unused(http, graph, dead, len(images))
        seed = self._seed(req)
        model = await self._model(http, graph)

        first = images[0]
        width, height, size, reshape = self._edit_size(req, first, custom)
        megapixels = self.edit_megapixels(req.orientation)
        if not custom and reshape:
            # 内置工作流：居中裁切到目标方向的尺寸（ImageScale 的 crop=center）
            graph["11"] = {"class_type": "ImageScale", "inputs": {
                "image": ["10", 0], "upscale_method": "lanczos", "width": width, "height": height,
                "crop": "center"}}

        refs = [await self.upload_image(http, image) for image in images]
        if not slots:
            for nid, ref in zip(loaders, refs):
                graph[nid]["inputs"]["image"] = ref
        flattened = not custom and add_alpha_flatten(graph, "10", first)  # 内置工作流的 LoadImage 是 #10

        strength = as_float(req.strength, 0.0)
        denoise = round(min(1.0, max(0.01, strength if 0 < strength <= 1 else DEFAULT_EDIT_DENOISE)), 2)
        values = self._values(req, model, seed, width, height)
        values.update(denoise=denoise, megapixels=megapixels, image=refs[0])
        for i in range(1, MAX_EDIT_IMAGES + 1):
            values[f"image{i}"] = refs[i - 1] if i <= len(refs) else ""
        used: set = set()
        graph = fill_placeholders(graph, values, used)
        model = apply_fallbacks(graph, values, used, model)  # 提示词连线、随机种子、Empty*Latent* 尺寸

        image, pid, warning = await self.run_graph(http, graph, extra)
        params = {
            "mode": "edit",
            "model": model,
            "size": size,
            "seed": seed,
            "input_images": len(refs),
            "workflow": "custom-edit" if custom else "builtin-img2img",
            "prompt_id": pid,
        }
        if "denoise" in used:
            params["strength"] = denoise
        if ignored:
            params["ignored_images"] = ignored
        if flattened:
            params["alpha"] = "white"
        params.update(self._sampler_params(values, used))
        note = f"修改工作流只使用前 {capacity} 张图片，其余 {ignored} 张已忽略" if ignored else ""
        return image_result(image, params, "；".join(text for text in (note, warning) if text))

    async def upload_image(self, http, image: InputImage) -> str:
        """上传一张输入图片，返回填入 LoadImage.image 的值。

        文件名取内容哈希且 overwrite=false：本地 ComfyUI 遇到同名同内容的文件直接复用，Comfy Cloud 按内容哈希存储，
        所以超时、5xx 后重试不会产生重复文件（上传本身也不生成图片、不计费）。
        """
        data = image.data
        mime = detect_mime(data) or as_str(image.mime) or "image/png"
        filename = f"multidraw_{hashlib.sha256(data).hexdigest()[:32]}{UPLOAD_EXT.get(mime, '.png')}"
        # type 必须写 input：Comfy Cloud 不写时默认 output
        form = {"type": "input", "overwrite": "false"}
        if not self.cloud:
            form["subfolder"] = UPLOAD_SUBFOLDER
        url = self.url("/api/upload/image" if self.cloud else "/upload/image")
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                resp = await send(http, "POST", url, label=self.name, proxy=self.proxy,
                                  timeout=min(120.0, self.timeout), headers=self.headers,
                                  files={"image": (filename, data, mime)}, data=form, on_error=self.map_error)
                break
            except DrawError as e:
                if e.status == 413:
                    raise DrawError(f"{e}。图片超过了 ComfyUI 的上传大小上限（默认 100 MB，启动参数 --max-upload-size）",
                                    status=413) from e
                if attempt >= UPLOAD_ATTEMPTS or not (e.retryable or e.status is None or e.status >= 500) \
                        or (e.retry_after or 0) > MAX_RETRY_WAIT:
                    raise
                await asyncio.sleep(e.retry_after if e.retry_after is not None else attempt)
        body = json_body(resp, self.name)
        name = as_str(body.get("name"))
        if not name:
            raise DrawError(f"{self.name} 上传图片后没有返回文件名：{response_detail(body)}")
        subfolder = as_str(body.get("subfolder")).strip("/\\")
        # 本地：LoadImage 的值是相对 input 目录的路径（与前端粘贴图片的 "pasted/xxx.png" 相同）；Cloud 只用 name
        return f"{subfolder}/{name}" if subfolder and not self.cloud else name

    async def required_inputs(self, http, classes: set[str]) -> dict[str, frozenset]:
        """节点类型 -> 必填输入名。本地按类型查 GET /object_info/{类型}，Cloud 只提供整表 GET /api/object_info。

        查不到的类型不在结果里（prune_nodes 对它们退回「没有连线就删」）；结果缓存 10 分钟。
        """
        now, base, out, todo = time.monotonic(), self.base_url, {}, []
        for cls in sorted(classes):
            hit = _REQUIRED_CACHE.get((base, cls))
            if hit and hit[0] > now:
                if hit[1] is not None:
                    out[cls] = hit[1]
            else:
                todo.append(cls)
        if not todo:
            return out
        try:
            if self.cloud:
                data = json_body(await self.api_get(http, "/api/object_info", attempts=1, timeout=60.0), self.name)
            else:
                data = {}
                for cls in todo:  # 不存在的类型返回 {}
                    data.update(json_body(await self.api_get(http, "/object_info/" + quote(cls, safe=""),
                                                             attempts=1), self.name))
        except DrawError:
            return out
        for cls in todo:
            required = required_from_info(data.get(cls))
            _REQUIRED_CACHE[(base, cls)] = (now + REQUIRED_CACHE_TTL, required)
            if required is not None:
                out[cls] = required
        return out

    async def prune_unused(self, http, graph: dict, dead: set, used_images: int) -> None:
        """删掉多余的图片加载节点及只为它们服务的下游节点；连输出节点都被删掉时说明原因。"""
        if not dead:
            return
        required = await self.required_inputs(http, downstream_classes(graph, dead))
        classes = {nid: str(node.get("class_type")) for nid, node in graph.items()}
        removed = prune_nodes(graph, dead, required)
        lost = sorted({classes[nid] for nid in removed if classes.get(nid) in OUTPUT_CLASSES})
        if lost and not any(str(node.get("class_type")) in OUTPUT_CLASSES for node in graph.values()):
            raise DrawError(f"这个图片修改工作流在只有 {used_images} 张图片时无法运行（多余的图片连到了必填输入，"
                            f"连 {lost[0]} 输出节点也用不上了），请多发几张图片或修改工作流")

    async def submit(self, http, body: dict, pid: str) -> tuple[str, str]:
        """提交工作流，返回 (prompt_id, 警告)。只在请求确定没有被处理时重试。"""
        try:
            resp = await with_retries(lambda: send(
                http, "POST", self.url("/api/prompt" if self.cloud else "/prompt"), label=self.name,
                proxy=self.proxy, timeout=min(60.0, self.timeout), headers=self.headers, json=body,
                on_error=self.map_error,
            ))
        except DrawError as e:
            if e.status is None and not e.retryable and pid and not self.cloud:
                # 超时或连接中断时任务可能已进入队列：按自己的 prompt_id 查找，找到就继续等待，避免重复提交
                try:
                    if await self.locate(http, pid, attempts=1):
                        return pid, ""
                except DrawError:
                    pass
            if e.status == 500 and not self.cloud:
                raise DrawError(f"{e}。如果使用了自定义工作流，请确认它是 API 格式，并且用到的自定义节点都已安装",
                                status=500) from e
            raise
        data = json_body(resp, self.name)
        new_pid = as_str(data.get("prompt_id"))
        if not new_pid:
            raise DrawError(f"{self.name} 没有返回 prompt_id：{response_detail(data)}")
        warning = ""
        if data.get("node_errors"):  # 部分输出节点未通过校验，其余已进入队列
            warning = "部分节点未通过校验：" + format_prompt_error({"node_errors": data["node_errors"]})
        return new_pid, warning

    async def locate(self, http, pid: str, *, attempts: int = 3) -> str:
        """任务位置：history / queue_running / queue_pending，找不到时返回空串。"""
        if pid in json_body(await self.api_get(http, f"/history/{pid}", attempts=attempts), self.name):
            return "history"
        queue = json_body(await self.api_get(http, "/queue", attempts=attempts), self.name)
        for key in ("queue_running", "queue_pending"):
            items = queue.get(key)
            for item in items if isinstance(items, list) else []:
                if isinstance(item, list) and len(item) > 1 and item[1] == pid:
                    return key
        return ""

    def timeout_error(self) -> DrawError:
        return DrawError(f"{self.name} 生成超时（{self.timeout:g} 秒，含排队和加载模型的时间），已尝试取消任务。"
                         "可以在配置中调大超时时间")

    async def wait_local(self, http, pid: str, deadline: float) -> dict:
        polls = misses = 0
        while True:
            history = json_body(await self.api_get(http, f"/history/{pid}"), self.name)
            entry = history.get(pid)
            if isinstance(entry, dict):
                status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise DrawError(format_history_error(status))
                return entry.get("outputs") if isinstance(entry.get("outputs"), dict) else {}
            polls += 1
            if polls % 5 == 0:
                # 任务结束前 history 一直是 {}：定期确认任务还在队列中，ComfyUI 重启后不会一直等到超时
                where = await self.locate(http, pid)
                if where == "history":
                    continue
                misses = 0 if where else misses + 1
                if misses >= 2:
                    raise DrawError("ComfyUI 任务丢失：既不在队列中也不在历史记录中（ComfyUI 可能重启了）")
            if time.monotonic() >= deadline:
                await self.cancel_quietly(http, pid)
                raise self.timeout_error()
            await asyncio.sleep(self.poll_interval)

    async def wait_cloud(self, http, pid: str, deadline: float) -> dict:
        while True:
            try:
                job = json_body(await self.api_get(http, f"/api/jobs/{pid}"), self.name)
            except DrawError as e:
                if e.status != 404:
                    raise
                job = {}  # 刚提交时可能暂时查不到
            status = as_str(job.get("status")).lower()
            if status in _CLOUD_DONE:
                outputs = dict(job.get("outputs") or {}) if isinstance(job.get("outputs"), dict) else {}
                preview = job.get("preview_output")
                if isinstance(preview, dict):
                    outputs.setdefault("preview_output", {"images": [preview]})
                return outputs
            if status in _CLOUD_FAILED:
                error = job.get("execution_error")
                if isinstance(error, dict) and error:
                    raise DrawError(format_exec_error(error, self.name))
                raise DrawError(f"{self.name} 任务失败（{status}）")
            if time.monotonic() >= deadline:
                await self.cancel_quietly(http, pid)
                raise self.timeout_error()
            await asyncio.sleep(self.cloud_poll_interval)

    async def cancel(self, http, pid: str) -> None:
        if self.cloud:
            # Cloud：/api/queue 只删除排队中的任务，/api/interrupt 中断该账号所有正在运行的任务
            try:
                job = json_body(await self.api_get(http, f"/api/jobs/{pid}", attempts=1), self.name)
                status = as_str(job.get("status")).lower()
            except DrawError:
                status = ""
            if status in _CLOUD_DONE or status in _CLOUD_FAILED:
                return
            await self.api_post(http, "/api/queue", {"delete": [pid]})
            if status == "in_progress":
                await self.api_post(http, "/api/interrupt")
            return
        try:
            # 新版：原子操作，排队中则移除，正在运行则只中断这个任务
            resp = await self.api_post(http, f"/api/jobs/{pid}/cancel")
            if "cancelled" in json_body(resp, self.name):
                return
        except DrawError:
            pass
        # 旧版：排队中的从队列删除；正在运行的按 prompt_id 中断（确认是自己的任务，避免中断别人的）
        where = await self.locate(http, pid, attempts=1)
        if where == "queue_pending":
            await self.api_post(http, "/queue", {"delete": [pid]})
        elif where == "queue_running":
            await self.api_post(http, "/interrupt", {"prompt_id": pid})

    async def cancel_quietly(self, http, pid: str) -> None:
        """尽力取消服务端任务，失败不影响后续流程；不会被外层的取消打断。"""
        try:
            await asyncio.shield(asyncio.wait_for(self.cancel(http, pid), 10))
        except Exception:  # noqa: BLE001 - 只是尽力而为
            pass

    async def fetch_image(self, http, image: dict) -> bytes:
        query = urlencode({
            "filename": as_str(image.get("filename")),
            "subfolder": as_str(image.get("subfolder")),
            "type": as_str(image.get("type")) or "output",
        })
        timeout = min(120.0, self.timeout)
        if not self.cloud:
            return await download_image(http, self.url(f"/view?{query}"), label=self.name, proxy=self.proxy,
                                        timeout=timeout, headers=self.headers)
        url = self.url(f"/api/view?{query}")
        try:
            resp = await send(http, "GET", url, label=self.name, proxy=self.proxy, timeout=timeout,
                              headers=self.headers, on_error=self.map_error, follow_redirects=False)
        except _Redirect as r:
            # 302 到签名的存储地址：下载时不能带 X-API-Key
            return await download_image(http, urljoin(url, r.location), label=self.name, proxy=self.proxy,
                                        timeout=timeout)
        if not detect_mime(resp.content):
            raise DrawError(f"{self.name} 返回的图片内容不是可识别的图片")
        return resp.content

    # ------------------------------------------------------------ 余额

    def balance_key(self) -> str:
        """查询积分余额用的 comfy.org Key：Comfy Cloud 用配置的 API Key；本地 ComfyUI 用额外请求参数中
        extra_data.api_key_comfy_org（付费 API 节点用的 Key），没有时返回空串。"""
        if self.cloud:
            return self.api_key
        try:
            extra = self.extra_body()
        except DrawError:
            return ""
        data = extra.get("extra_data")
        return as_str(data.get("api_key_comfy_org")) if isinstance(data, dict) else ""

    def balance_missing(self) -> str:
        if self.cloud:
            return "" if self.api_key else "管理员还没有配置 Comfy Cloud API Key"
        if not self.balance_key():
            return ("本地 ComfyUI 不计费，没有余额可查（只有使用 Comfy Cloud，或在「额外请求参数」的 "
                    "extra_data.api_key_comfy_org 中填写了 comfy.org API Key 来使用付费 API 节点时才有积分余额）")
        return ""

    async def balance(self, clients: Clients) -> BalanceInfo:
        reason = self.balance_missing()
        if reason:
            raise DrawError(reason)
        http = clients.http
        if self.cloud:
            # 官方文档：GET /api/billing/usage/timeseries（X-API-Key）。summary.balance 的单位是「分」，
            # 读不到余额时省略；summary.spend_micros 的单位是百万分之一美元，数据来自账单
            try:
                resp = await self.api_get(http, "/api/billing/usage/timeseries?granularity=month&months=1",
                                          attempts=2)
                data = json_body(resp, self.name)
                summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
                balance = summary.get("balance")
                if isinstance(balance, dict) and balance:
                    spend = _number(summary.get("spend_micros"))
                    return BalanceInfo(
                        items=balance_items(balance, spend_usd=None if spend is None else spend / 1_000_000),
                        note="消费数据来自账单，最近的用量可能尚未计入",
                    )
            except DrawError as e:
                # 400 / 401 / 402 / 429 是 Key、订阅或参数的问题，换接口也不会好；
                # 没有该接口（404/405）、5xx、网络错误或数据格式不对时改用 api.comfy.org（同一个 Key）
                if not (e.status in (404, 405) or e.status is None or e.status >= 500):
                    raise
        key = self.balance_key()
        try:
            resp = await with_retries(lambda: send(
                http, "GET", f"{COMFY_PLATFORM_API}/customers/balance", label="Comfy 平台", proxy=self.proxy,
                timeout=30.0, headers={"X-API-Key": key}))
        except DrawError as e:
            if e.status == 404:  # 官方文档：404 Customer not found
                raise DrawError("这个 comfy.org 账号还没有积分账户（从未购买或领取过积分）", status=404) from e
            if e.status in (401, 403) and not self.cloud:
                raise DrawError(f"{e}。查询余额使用的是「额外请求参数」中 extra_data.api_key_comfy_org 的 Key",
                                status=e.status) from e
            raise
        note = "Comfy Cloud 与付费 API 节点共用同一个积分余额"
        if self.cloud:  # /customers/balance 是账号本人的余额；团队 Workspace 的共享余额只在 timeseries 里
            note += "；使用 Cloud 团队 Workspace 时这里显示的是个人账号的余额，团队余额请在 Plan & Credits 中查看"
        return BalanceInfo(items=balance_items(json_body(resp, "Comfy 平台")), note=note)
