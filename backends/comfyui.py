"""ComfyUI 后端：本地 / 自部署 ComfyUI 的文生图，也支持 Comfy Cloud（https://cloud.comfy.org）。

本地流程：POST /prompt 提交 API 格式工作流 → 轮询 GET /history/{prompt_id} → GET /view 下载第一张输出图片。
Comfy Cloud：POST /api/prompt（X-API-Key）→ 轮询 GET /api/jobs/{id} → GET /api/view（302 到签名地址，下载时不带 Key）。
"""

import asyncio
import copy
import ipaddress
import json
import math
import random
import re
import time
import uuid
from urllib.parse import urlencode, urljoin, urlparse

import httpx

from .base import (
    Backend,
    Clients,
    DrawError,
    DrawRequest,
    DrawResult,
    as_float,
    as_int,
    as_str,
    detect_mime,
    download_image,
    error_from_response,
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

_PH = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
_PH_WHOLE = re.compile(r"^\s*\{\{\s*([A-Za-z_]\w*)\s*\}\}\s*$")
_BILLING = re.compile(r"quota|billing|credit|balance|insufficient|payment|paid|subscription|余额|额度|欠费|充值", re.I)
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


def load_workflow(raw) -> dict:
    """解析 API 格式的工作流；留空时返回内置的 SDXL 工作流。"""
    if isinstance(raw, dict):
        data = copy.deepcopy(raw)
    else:
        text = as_str(raw).lstrip("\ufeff")  # 去掉 BOM
        if not text:
            return copy.deepcopy(DEFAULT_WORKFLOW)
        try:
            data = json.loads(_quote_bare_placeholders(text))
        except ValueError as e:  # 包括 JSONDecodeError 和超长数字
            raise DrawError(f"ComfyUI 工作流不是有效的 JSON：{e}") from e
    if not data:
        return copy.deepcopy(DEFAULT_WORKFLOW)
    if isinstance(data, dict) and _is_api_graph(data.get("prompt")):
        data = data["prompt"]  # 兼容 {"prompt": {...}} 形式
    if isinstance(data, dict) and ("nodes" in data or "links" in data):
        raise DrawError(UI_FORMAT_ERROR)
    if not _is_api_graph(data):
        bad = [str(k) for k, v in data.items() if not (isinstance(v, dict) and "class_type" in v)] \
            if isinstance(data, dict) else []
        where = f"（异常的键：{', '.join(bad[:5])}）" if bad else ""
        raise DrawError(f"ComfyUI 工作流不是 API 格式：每个顶层的值都应是含 class_type 和 inputs 的节点{where}。"
                        "请用「导出 (API)」/ Save (API Format) 导出")
    unknown = sorted(find_placeholders(data) - set(PLACEHOLDERS))
    if unknown:
        raise DrawError(f"ComfyUI 工作流中有未知的占位符 {{{{{unknown[0]}}}}}，可用：{', '.join(PLACEHOLDERS)}")
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


# ---------------------------------------------------------------- 后端


class ComfyUIBackend(Backend):
    key = "comfyui"
    label = "ComfyUI"
    # 内置工作流面向 SDXL 动漫模型（Illustrious / NoobAI 等），使用 Danbooru 标签
    default_prompt_mode = "tags"
    supports_negative = True
    supports_models = True
    doc_url = "https://docs.comfy.org/development/comfyui-server/comms_routes"
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

    async def _generate(self, http, req: DrawRequest) -> DrawResult:
        graph = load_workflow(self.section.get("workflow_json"))
        extra = self.extra_body()  # 先解析，配置有误时不发出请求
        width, height = self.size(req.orientation)
        seed = req.seed if isinstance(req.seed, int) and 0 < req.seed <= MAX_SEED else random.randint(1, 2**32 - 1)
        model = self.model
        if not model and find_placeholders(graph) & {"model", "ckpt_name"}:
            model = await self.default_model(http)
        values = {
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
        used: set = set()
        graph = fill_placeholders(graph, values, used)
        model = apply_fallbacks(graph, values, used, model)

        body = {"prompt": graph}
        if not self.cloud:
            # 客户端生成 prompt_id（新版要求标准 UUID）：提交超时或取消时可以按它查找、取消任务
            body.update(client_id=uuid.uuid4().hex, prompt_id=str(uuid.uuid4()))
        body.update(extra)
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
        image = await self.fetch_image(http, pick_image(outputs))

        params = {
            "model": model,
            "size": f"{width}x{height}",
            "seed": seed,
            "workflow": "custom" if self.custom_workflow else "builtin",
            "prompt_id": pid,
        }
        names = {"steps": "steps", "cfg": "cfg", "sampler_name": "sampler", "scheduler": "scheduler"}
        params.update({names[k]: values[k] for k in names if k in used})
        return image_result(image, params, warning)

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
