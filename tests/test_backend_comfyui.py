"""ComfyUI 后端的单元测试（模拟 ComfyUI / Comfy Cloud，不访问网络）：python -m pytest tests"""

import asyncio
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, unquote

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import comfyui  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.comfyui import ComfyUIBackend  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG = b"\xff\xd8\xff\xe0" + b"\x01" * 32
REAL_SLEEP = asyncio.sleep
CHECKPOINTS = ["illustriousXL_v01.safetensors", "sd_xl_base_1.0.safetensors"]
OK_ENTRY = {
    "outputs": {
        "8": {"images": [{"filename": "preview_00001_.png", "subfolder": "", "type": "temp"}]},
        "9": {"images": [{"filename": "draw_00001_.png", "subfolder": "astrbot", "type": "output",
                          "metadata": {"x": 1}}]},
    },
    "status": {"status_str": "success", "completed": True,
               "messages": [["execution_start", {}], ["execution_success", {}]]},
}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """去掉重试和轮询的等待，并避免环境变量中的代理影响模拟请求。"""
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: REAL_SLEEP(0))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    comfyui._REQUIRED_CACHE.clear()  # /object_info 的缓存不能影响其他测试
    yield
    comfyui._REQUIRED_CACHE.clear()


def parse_multipart(request: httpx.Request) -> tuple[dict, dict]:
    """解析 multipart/form-data 请求体：返回 (普通字段, {字段名: (文件名, Content-Type, 内容)})。"""
    boundary = request.headers["content-type"].split("boundary=", 1)[1].strip('"').encode()
    fields, files = {}, {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, data = chunk.removeprefix(b"\r\n").partition(b"\r\n\r\n")
        data = data.removesuffix(b"\r\n")
        head = head.decode()
        name = re.search(r'\bname="([^"]*)"', head).group(1)
        filename = re.search(r'\bfilename="([^"]*)"', head)
        if filename:
            ctype = re.search(r"content-type:\s*(\S+)", head, re.IGNORECASE)
            files[name] = (filename.group(1), ctype.group(1) if ctype else "", data)
        else:
            fields[name] = data.decode()
    return fields, files


def _info(required, optional=()):
    return {"input": {"required": {k: ["X"] for k in required}, "optional": {k: ["X"] for k in optional}}}


# ComfyUI 的 /object_info 中真实节点的必填 / 可选输入（节选）
OBJECT_INFO = {
    "ImageScale": _info(["image", "upscale_method", "width", "height", "crop"]),
    "ImageScaleToTotalPixels": _info(["image", "upscale_method", "megapixels", "resolution_steps"]),
    "VAEEncode": _info(["pixels", "vae"]),
    "ReferenceLatent": _info(["conditioning"], ["latent"]),
    "TextEncodeQwenImageEditPlus": _info(["clip", "prompt"], ["vae", "image1", "image2", "image3"]),
    "KSampler": _info(["model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative",
                       "latent_image", "denoise"]),
    "VAEDecode": _info(["samples", "vae"]),
    "SaveImage": _info(["images", "filename_prefix"]),
    "GetImageSize": _info(["image"]),
    "EmptyLatentImage": _info(["width", "height", "batch_size"]),
}


class FakeClock:
    """每次读取前进 100 秒，用来触发总超时。"""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        self.now += 100
        return self.now


class FakeComfy:
    """模拟本地 ComfyUI：记录请求，提交的任务在 pending 次轮询后完成。"""

    def __init__(self, *, pending=1, entry=None, checkpoints=CHECKPOINTS, on_prompt=None, in_queue=False,
                 cancel_api=True, view=PNG, upload_fail=(), object_info_status=200):
        self.pending = pending
        self.entry = OK_ENTRY if entry is None else entry
        self.checkpoints = checkpoints
        self.on_prompt = on_prompt
        self.in_queue = in_queue
        self.cancel_api = cancel_api
        self.view = view
        self.upload_fail = list(upload_fail)  # 上传图片前几次返回的状态码
        self.object_info_status = object_info_status
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.uploads: list[tuple[dict, dict]] = []
        self.accepted: list[str] = []
        self.polls = 0
        self.object_info_calls = 0

    def paths(self, method=None):
        return [r.url.path for r in self.requests if method is None or r.method == method]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path
        if method == "POST" and path == "/prompt":
            body = json.loads(request.content)
            self.bodies.append(body)
            if self.on_prompt is not None:
                resp = self.on_prompt(request, body)
                if resp is not None:
                    return resp
            pid = body.get("prompt_id") or "server-pid"
            self.accepted.append(pid)
            return httpx.Response(200, json={"prompt_id": pid, "number": 1, "node_errors": {}})
        if method == "GET" and path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            self.polls += 1
            if pid not in self.accepted or self.polls <= self.pending:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={pid: self.entry})
        if method == "GET" and path == "/queue":
            items = [[1, pid, {}, {}, []] for pid in self.accepted] if self.in_queue else []
            key = "queue_pending" if self.in_queue == "pending" else "queue_running"
            return httpx.Response(200, json={"queue_running": [], "queue_pending": [], key: items})
        if method == "GET" and path == "/view":
            return httpx.Response(200, content=self.view, headers={"Content-Type": "image/png"})
        if method == "GET" and path == "/models/checkpoints":
            return httpx.Response(200, json=list(self.checkpoints))
        if method == "POST" and path.startswith("/api/jobs/") and path.endswith("/cancel") and self.cancel_api:
            return httpx.Response(200, json={"cancelled": True})
        if method == "POST" and path in ("/queue", "/interrupt"):
            return httpx.Response(200)
        if method == "POST" and path == "/upload/image":
            fields, files = parse_multipart(request)
            self.uploads.append((fields, files))
            if self.upload_fail:
                return httpx.Response(self.upload_fail.pop(0), text="")
            return httpx.Response(200, json={"name": files["image"][0], "subfolder": fields.get("subfolder", ""),
                                             "type": fields.get("type", "input")})
        if method == "GET" and path.startswith("/object_info/"):
            self.object_info_calls += 1
            if self.object_info_status != 200:
                return httpx.Response(self.object_info_status, text="boom")
            cls = unquote(path.split("/", 2)[2])
            return httpx.Response(200, json={cls: OBJECT_INFO[cls]} if cls in OBJECT_INFO else {})
        return httpx.Response(404, json={"error": "Not Found"})


def run(section, handler, req=None, *, action="generate"):
    async def go():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            backend = ComfyUIBackend(section)
            if action == "models":
                return await backend.list_models(Clients(http=pool))
            if action == "balance":
                return await backend.balance(Clients(http=pool))
            return await backend.generate(Clients(http=pool), req or DrawRequest("1girl, smile", "lowres", seed=42))
        finally:
            await pool.close()

    return asyncio.run(go())


def graph_of(fake: FakeComfy) -> dict:
    assert len(fake.bodies) == 1
    return fake.bodies[0]["prompt"]


# ---------------------------------------------------------------- 内置工作流


@pytest.mark.parametrize(("orientation", "size"),
                         [("portrait", (832, 1216)), ("landscape", (1216, 832)), ("square", (1024, 1024)),
                          ("unknown", (832, 1216)), (None, (832, 1216))])
def test_default_workflow_request(orientation, size):
    fake = FakeComfy()
    result = run({}, fake, DrawRequest('a "quoted" \\ prompt\nline2', "lowres, bad hands", orientation, seed=42))
    body = fake.bodies[0]
    graph = body["prompt"]
    assert fake.paths() == ["/models/checkpoints", "/prompt", "/history/" + body["prompt_id"],
                            "/history/" + body["prompt_id"], "/view"]
    assert str(uuid.UUID(body["prompt_id"])) == body["prompt_id"] and body["client_id"]
    sampler = graph["3"]["inputs"]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == \
        (42, 25, 7.0, "dpmpp_2m", "karras")
    assert graph["4"]["inputs"]["ckpt_name"] == CHECKPOINTS[0]  # 模型留空：用列表中的第一个
    assert (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]) == size
    assert graph["6"]["inputs"]["text"] == 'a "quoted" \\ prompt\nline2'
    assert graph["7"]["inputs"]["text"] == "lowres, bad hands"
    assert graph["3"]["inputs"]["positive"] == ["6", 0] and graph["9"]["class_type"] == "SaveImage"
    assert all("authorization" not in r.headers and "x-api-key" not in r.headers for r in fake.requests)
    # 取 SaveImage 的输出（type=output），而不是预览图
    query = parse_qs(fake.requests[-1].url.query.decode(), keep_blank_values=True)
    assert query == {"filename": ["draw_00001_.png"], "subfolder": ["astrbot"], "type": ["output"]}
    assert result.image == PNG and result.mime == "image/png" and result.text == ""
    assert result.params == {
        "model": CHECKPOINTS[0], "size": f"{size[0]}x{size[1]}", "seed": 42, "workflow": "builtin",
        "prompt_id": body["prompt_id"], "steps": 25, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
    }


def test_configured_model_sd15_sizes_and_sampler():
    fake = FakeComfy()
    section = {"model": " v1-5-pruned-emaonly.safetensors ", "portrait_size": "512x768", "steps": "30",
               "cfg": "5.5", "sampler_name": "euler_ancestral", "scheduler": "normal"}
    result = run(section, fake, DrawRequest("cat", seed=0))
    graph = graph_of(fake)
    assert "/models/checkpoints" not in fake.paths()
    assert graph["4"]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly.safetensors"
    assert (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]) == (512, 768)
    inputs = graph["3"]["inputs"]
    assert (inputs["steps"], inputs["cfg"], inputs["sampler_name"], inputs["scheduler"]) == \
        (30, 5.5, "euler_ancestral", "normal")
    assert 1 <= inputs["seed"] <= 2**32 - 1 and result.params["seed"] == inputs["seed"]  # seed=0 → 随机
    assert graph["7"]["inputs"]["text"] == ""  # 没有负面提示词


def test_random_seed_changes_between_runs():
    seeds = set()
    for _ in range(3):
        fake = FakeComfy()
        run({"model": "m.safetensors"}, fake, DrawRequest("cat"))
        seeds.add(graph_of(fake)["3"]["inputs"]["seed"])
    assert len(seeds) == 3


# ---------------------------------------------------------------- 自定义工作流


FLUX_WORKFLOW = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5xxl_fp16.safetensors",
                                                      "clip_name2": "clip_l.safetensors", "type": "flux"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["2", 0]}},
    "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": 5}},
    "6": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": "{{steps}}", "denoise": 1,
                                                      "model": ["1", 0]}},
    "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["5", 0], "sigmas": ["6", 0],
                                                             "latent_image": ["4", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["9", 0]}},
    "9": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "flux", "images": ["8", 0]}},
}


def test_custom_flux_workflow_without_negative():
    fake = FakeComfy()
    section = {"workflow_json": json.dumps(FLUX_WORKFLOW), "prompt_mode": "natural", "steps": 20}
    assert ComfyUIBackend(section).prompt_mode == "natural"
    result = run(section, fake, DrawRequest("a cat on a sofa", "blurry", "landscape", seed=777))
    graph = graph_of(fake)
    assert "/models/checkpoints" not in fake.paths()  # 没有 {{model}}，不查询模型
    assert graph["3"]["inputs"]["text"] == "a cat on a sofa"
    assert graph["5"]["inputs"]["noise_seed"] == 777  # 没有 {{seed}}：固定种子被替换
    assert (graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]) == (1216, 832)
    assert graph["6"]["inputs"]["steps"] == 20 and graph["1"]["inputs"]["unet_name"] == "flux1-dev.safetensors"
    assert "blurry" not in json.dumps(graph)  # 工作流没有负面提示词，不强行写入
    assert result.params["workflow"] == "custom" and result.params["model"] == ""
    assert result.params["steps"] == 20 and "cfg" not in result.params and result.params["size"] == "1216x832"

    # 配置了模型但工作流用不到（UNETLoader）：不把没用上的模型写进记录
    fake = FakeComfy()
    result = run({**section, "model": "sd_xl_base_1.0.safetensors"}, fake, DrawRequest("cat", seed=1))
    assert result.params["model"] == "" and "sd_xl_base" not in json.dumps(graph_of(fake))


def test_placeholders_are_typed_and_escaped():
    workflow = """{
      "3": {"class_type": "KSampler", "inputs": {"seed": {{seed}}, "steps": "{{steps}}", "cfg": "{{ cfg }}",
            "sampler_name": "{{sampler_name}}", "scheduler": "{{scheduler}}", "denoise": 1,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
      "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "{{ckpt_name}}"}},
      "5": {"class_type": "EmptyLatentImage", "inputs": {"width": "{{width}}", "height": {{height}}, "batch_size": 1}},
      "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece, {{prompt}}, literal", "clip": ["4", 1]}},
      "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{NEGATIVE_PROMPT}}", "clip": ["4", 1]}},
      "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "seed {{seed}}", "images": ["6", 0]}}
    }"""
    fake = FakeComfy()
    prompt = 'say "hi" \\ {{seed}}\n}'
    run({"workflow_json": workflow, "model": "a/b.safetensors", "cfg": 6}, fake, DrawRequest(prompt, "ugly", seed=9))
    graph = graph_of(fake)
    inputs = graph["3"]["inputs"]
    assert inputs["seed"] == 9 and isinstance(inputs["steps"], int) and inputs["cfg"] == 6.0
    assert isinstance(inputs["cfg"], float)
    assert graph["5"]["inputs"]["width"] == 832 and graph["5"]["inputs"]["height"] == 1216
    assert graph["4"]["inputs"]["ckpt_name"] == "a/b.safetensors"
    # 提示词中的引号、反斜杠、换行和 {{...}} 原样保留，不会被再次替换
    assert graph["6"]["inputs"]["text"] == f"masterpiece, {prompt}, literal"
    assert graph["7"]["inputs"]["text"] == "ugly"
    assert graph["9"]["inputs"]["filename_prefix"] == "seed 9"


def test_api_export_without_placeholders_is_auto_wired():
    export = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 123, "steps": 20, "cfg": 8, "sampler_name": "euler",
                                                    "scheduler": "normal", "denoise": 1, "model": ["4", 0],
                                                    "positive": ["6", 0], "negative": ["7", 0],
                                                    "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "foo.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old positive", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative", "clip": ["4", 1]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["3", 0]}},
    }
    fake = FakeComfy()
    result = run({"workflow_json": {"prompt": export}, "model": "bar.safetensors"}, fake,
                 DrawRequest("1girl", "lowres", "square", seed=777))
    graph = graph_of(fake)
    assert graph["6"]["inputs"]["text"] == "1girl" and graph["7"]["inputs"]["text"] == "lowres"
    assert graph["3"]["inputs"]["seed"] == 777 and graph["3"]["inputs"]["steps"] == 20
    assert (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]) == (1024, 1024)
    assert graph["4"]["inputs"]["ckpt_name"] == "bar.safetensors"  # 只有一个 ckpt_name 时替换为配置的模型
    assert result.params["model"] == "bar.safetensors" and "steps" not in result.params
    assert export["6"]["inputs"]["text"] == "old positive"  # 不修改配置中的对象

    fake = FakeComfy()
    result = run({"workflow_json": json.dumps(export)}, fake, DrawRequest("1girl", "", seed=5))
    graph = graph_of(fake)
    assert graph["7"]["inputs"]["text"] == "old negative"  # 没有负面提示词时保留原来的
    assert graph["4"]["inputs"]["ckpt_name"] == "foo.safetensors" and result.params["model"] == "foo.safetensors"
    assert "/models/checkpoints" not in fake.paths()


def test_two_checkpoint_loaders_are_left_alone():
    workflow = json.loads(json.dumps(comfyui.DEFAULT_WORKFLOW))
    workflow["4"]["inputs"]["ckpt_name"] = "base.safetensors"
    workflow["11"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "refiner.safetensors"}}
    fake = FakeComfy()
    result = run({"workflow_json": json.dumps(workflow), "model": "other.safetensors"}, fake)
    graph = graph_of(fake)
    assert graph["4"]["inputs"]["ckpt_name"] == "base.safetensors"
    assert graph["11"]["inputs"]["ckpt_name"] == "refiner.safetensors"
    assert result.params["model"] == ""


@pytest.mark.parametrize(("workflow", "expected"), [
    ('{"last_node_id": 9, "last_link_id": 9, "nodes": [], "links": [], "version": 0.4}', "Save (API Format)"),
    ('{"3": {"class_type": "KSampler"', "不是有效的 JSON"),
    ('{"3": 5, "4": {"class_type": "SaveImage", "inputs": {}}}', "不是 API 格式"),
    ("[1, 2]", "不是 API 格式"),
    ('{"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{foo}}"}}}', "未知的占位符 {{foo}}"),
    # 模型留空且用了 {{model}}：未知占位符也要在查询模型列表之前报错
    ('{"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "{{model}}"}},'
     ' "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}} {{Denoise}}"}}}', "未知的占位符 {{denoise}}"),
    ('{"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}', "{{prompt}}"),
])
def test_workflow_errors_before_any_request(workflow, expected):
    fake = FakeComfy()
    with pytest.raises(DrawError) as info:
        run({"workflow_json": workflow}, fake)
    assert expected in str(info.value) and fake.requests == []


def test_missing_reports_bad_workflow():
    assert "Save (API Format)" in ComfyUIBackend({"workflow_json": '{"nodes": [], "links": []}'}).missing()
    assert "JSON" in ComfyUIBackend({"workflow_json": "{"}).missing()
    unknown = '{"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}, {{style}}"}}}'
    assert "未知的占位符 {{style}}" in ComfyUIBackend({"workflow_json": unknown}).missing()


# ---------------------------------------------------------------- 地址 / 配置


@pytest.mark.parametrize(("url", "expected", "cloud"), [
    ("", "http://127.0.0.1:8188", False),
    (None, "http://127.0.0.1:8188", False),
    ("  http://127.0.0.1:8188/  ", "http://127.0.0.1:8188", False),
    ("192.168.1.5:8188", "http://192.168.1.5:8188", False),
    ("localhost:8188", "http://localhost:8188", False),
    ("comfyui:8188", "http://comfyui:8188", False),  # Docker 服务名
    ("host.docker.internal:8188", "http://host.docker.internal:8188", False),
    ("comfy.example.com/", "https://comfy.example.com", False),
    ("https://user:pass@comfy.example.com/sub", "https://user:pass@comfy.example.com/sub", False),
    ("cloud.comfy.org/api/", "https://cloud.comfy.org", True),
    ("https://cloud.comfy.org", "https://cloud.comfy.org", True),
    ("http://cloud.comfy.org/api", "https://cloud.comfy.org", True),  # Cloud 只有 https
    # IP 地址（如 Tailscale 的 100.x、公网 IP）没有证书，按 http 处理
    ("100.100.1.2:8188", "http://100.100.1.2:8188", False),
    ("8.8.8.8:8188", "http://8.8.8.8:8188", False),
    ("[::1]:8188", "http://[::1]:8188", False),
    # 从浏览器地址栏复制时带上的参数和锚点
    ("http://127.0.0.1:8188/?workflow=a#/b", "http://127.0.0.1:8188", False),
    ("127.0.0.1:8188/#/", "http://127.0.0.1:8188", False),
])
def test_base_url_normalisation(url, expected, cloud):
    backend = ComfyUIBackend({"base_url": url})
    assert backend.base_url == expected and backend.cloud is cloud


@pytest.mark.parametrize("url", ["http://[bad", "http://127.0.0.1:abc", "127.0.0.1:99999", "http://　host:8188"])
def test_invalid_base_url_is_reported_without_crash(url):
    # 端口不是数字等地址会让 httpx 抛出 InvalidURL（不是 DrawError），必须在发请求前拦下
    assert "地址格式不正确" in ComfyUIBackend({"base_url": url}).missing()
    fake = FakeComfy()
    with pytest.raises(DrawError) as info:
        run({"base_url": url, "model": "m"}, fake)
    assert "地址格式不正确" in str(info.value) and fake.requests == []
    with pytest.raises(DrawError) as info:
        run({"base_url": url}, fake, action="models")
    assert "地址格式不正确" in str(info.value) and fake.requests == []
    ComfyUIBackend({"base_url": url}).summary()  # 状态查询不能出错


def test_requests_use_normalised_base_url():
    fake = FakeComfy()
    run({"base_url": "localhost:8188/", "model": "m.safetensors"}, fake)
    assert all(str(r.url).startswith("http://localhost:8188/") for r in fake.requests)


@pytest.mark.parametrize(("text", "default", "expected"), [
    ("832x1216", "1x1", (832, 1216)),
    (" 830 X 1222 ", "1x1", (832, 1224)),
    ("1024×1024", "1x1", (1024, 1024)),
    ("3:4", "1x1", (896, 1152)),
    ("16:9", "1x1", (1344, 768)),
    ("99999x10", "1x1", (16384, 16)),
    ("garbage", "1216x832", (1216, 832)),
    ("0x0", "1024x1024", (1024, 1024)),
    ("2048 x 2048", "1x1", (2048, 2048)),
    ("2K", "832x1216", (832, 1216)),
    ("", "1216x832", (1216, 832)),
    ("-512x512", "1024x1024", (1024, 1024)),
])
def test_fit_size(text, default, expected):
    assert comfyui.fit_size(text, default) == expected


def test_garbage_config_values_fall_back_to_defaults():
    section = {
        "api_key": None, "base_url": None, "proxy_url": None, "model": None, "portrait_size": "garbage",
        "landscape_size": None, "square_size": 123, "workflow_json": None, "steps": "", "cfg": "abc",
        "sampler_name": None, "scheduler": "", "prompt_mode": "weird", "timeout": "abc", "extra_body": None,
        "required_positive": None,
    }
    backend = ComfyUIBackend(section)
    assert backend.missing() == "" and backend.prompt_mode == "tags" and backend.timeout == 300
    assert (backend.steps, backend.cfg, backend.sampler, backend.scheduler) == (25, 7.0, "dpmpp_2m", "karras")
    assert backend.size("square") == (1024, 1024) and backend.size("landscape") == (1216, 832)
    fake = FakeComfy()
    run(section, fake)
    inputs = graph_of(fake)["3"]["inputs"]
    assert (inputs["steps"], inputs["cfg"], inputs["sampler_name"], inputs["scheduler"]) == \
        (25, 7.0, "dpmpp_2m", "karras")
    fake = FakeComfy()
    run(None, fake)  # 整个配置缺失
    assert graph_of(fake)["5"]["inputs"]["width"] == 832
    out_of_range = ComfyUIBackend({"steps": -5, "cfg": 1000, "timeout": 5})
    assert (out_of_range.steps, out_of_range.cfg, out_of_range.timeout) == (1, 100.0, 10.0)


def test_extra_body_is_merged_last():
    fake = FakeComfy()
    extra = '{"front": true, "client_id": "mine", "extra_data": {"api_key_comfy_org": "k"}}'
    run({"extra_body": extra, "model": "m.safetensors"}, fake)
    body = fake.bodies[0]
    assert body["front"] is True and body["client_id"] == "mine"
    assert body["extra_data"] == {"api_key_comfy_org": "k"} and "prompt" in body
    fake = FakeComfy()
    run({"extra_body": {"number": 3}, "model": "m.safetensors"}, fake)  # 也接受已经是对象的配置
    assert fake.bodies[0]["number"] == 3


@pytest.mark.parametrize("extra", ["[1, 2]", "{bad json"])
def test_invalid_extra_body_sends_nothing(extra):
    fake = FakeComfy()
    with pytest.raises(DrawError) as info:
        run({"extra_body": extra}, fake)
    assert "额外请求参数" in str(info.value) and fake.requests == []


def test_missing_and_summary():
    assert ComfyUIBackend({}).missing() == ""
    assert ComfyUIBackend(None).missing() == ""
    assert "Comfy Cloud API Key" in ComfyUIBackend({"base_url": "https://cloud.comfy.org"}).missing()
    assert ComfyUIBackend({"base_url": "https://cloud.comfy.org", "api_key": "k"}).missing() == ""
    assert ComfyUIBackend({}).summary() == "ComfyUI / 自动选择模型 / 竖图 832x1216 / 25 步"
    summary = ComfyUIBackend({"model": "x.safetensors", "portrait_size": "3:4",
                              "workflow_json": json.dumps(FLUX_WORKFLOW)}).summary()
    assert summary == "ComfyUI / x.safetensors / 竖图 896x1152 / 自定义工作流"
    backend = ComfyUIBackend({})
    assert backend.key == "comfyui" and backend.label == "ComfyUI" and backend.supports_models
    assert backend.supports_negative and backend.max_concurrency is None and backend.doc_url.startswith("https://")
    # 基类的 get(name, default) 是读取配置的接口，不能被 HTTP 方法覆盖
    assert ComfyUIBackend({"sampler_name": " euler "}).get("sampler_name") == "euler"
    assert ComfyUIBackend({}).get("scheduler", "karras") == "karras"


# ---------------------------------------------------------------- 模型列表


def test_list_models_local():
    fake = FakeComfy(checkpoints=["a.safetensors", "sub\\b.safetensors"])
    assert run({}, fake, action="models") == ["a.safetensors", "sub\\b.safetensors"]
    assert fake.paths() == ["/models/checkpoints"]


def test_list_models_unsupported_and_empty():
    def handler(request):
        return httpx.Response(404, text="404: Not Found")

    with pytest.raises(DrawError) as info:
        run({}, handler, action="models")
    assert info.value.status == 404 and "模型文件名" in str(info.value)

    fake = FakeComfy(checkpoints=[])
    with pytest.raises(DrawError) as info:
        run({}, fake)
    assert "checkpoints 目录中没有模型" in str(info.value) and "/prompt" not in fake.paths()


# ---------------------------------------------------------------- 错误与重试


def test_validation_error_is_shown_per_node():
    def on_prompt(request, body):
        return httpx.Response(400, json={
            "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                      "details": "", "extra_info": {}},
            "node_errors": {"4": {"errors": [{
                "type": "value_not_in_list", "message": "Value not in list",
                "details": "ckpt_name: 'nope.safetensors' not in ['a.safetensors']",
                "extra_info": {"input_name": "ckpt_name"}}], "dependent_outputs": ["9"],
                "class_type": "CheckpointLoaderSimple"}},
        })

    fake = FakeComfy(on_prompt=on_prompt)
    with pytest.raises(DrawError) as info:
        run({"model": "nope.safetensors"}, fake)
    message = str(info.value)
    assert info.value.status == 400 and not info.value.retryable
    assert "Prompt outputs failed validation" in message and "节点 #4 CheckpointLoaderSimple.ckpt_name" in message
    assert "nope.safetensors" in message and fake.paths("POST") == ["/prompt"]


def test_missing_custom_node_hint():
    def on_prompt(request, body):
        return httpx.Response(400, json={"error": {
            "type": "missing_node_type", "message": "Node 'Foo' not found. The custom node may not be installed.",
            "details": "Node ID '#12'", "extra_info": {"node_id": "12", "class_type": "Foo"}}, "node_errors": {}})

    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(on_prompt=on_prompt))
    assert "Node 'Foo' not found" in str(info.value) and "自定义节点" in str(info.value)


def test_server_500_is_not_retried():
    fake = FakeComfy(on_prompt=lambda request, body: httpx.Response(500, text="500 Internal Server Error"))
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    assert info.value.status == 500 and "API 格式" in str(info.value) and fake.paths("POST") == ["/prompt"]


def test_proxy_auth_header_and_401():
    fake = FakeComfy()
    run({"api_key": "secret", "model": "m"}, fake)
    assert {r.headers.get("authorization") for r in fake.requests} == {"Bearer secret"}
    fake = FakeComfy()
    run({"api_key": "Basic dXNlcjpwYXNz", "model": "m"}, fake)
    assert {r.headers.get("authorization") for r in fake.requests} == {"Basic dXNlcjpwYXNz"}

    fake = FakeComfy(on_prompt=lambda request, body: httpx.Response(401, text="Unauthorized"))
    with pytest.raises(DrawError) as info:
        run({"api_key": "wrong", "model": "m"}, fake)
    assert info.value.status == 401 and "API Key" in str(info.value) and fake.paths("POST") == ["/prompt"]


def test_rate_limit_429_is_retried_once():
    calls = []

    def on_prompt(request, body):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="Too Many Requests", headers={"Retry-After": "0"})
        return None

    fake = FakeComfy(on_prompt=on_prompt)
    result = run({"model": "m"}, fake)
    assert result.image == PNG and fake.paths("POST") == ["/prompt", "/prompt"]

    fake = FakeComfy(on_prompt=lambda request, body: httpx.Response(429, text="Too Many Requests"))
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    assert info.value.status == 429 and fake.paths("POST") == ["/prompt", "/prompt"]  # 最多重试一次


def test_billing_429_is_not_retried():
    fake = FakeComfy(on_prompt=lambda request, body: httpx.Response(
        429, json={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}}))
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    assert info.value.status == 429 and not info.value.retryable and "quota" in str(info.value)
    assert fake.paths("POST") == ["/prompt"]


def test_submit_timeout_is_not_retried():
    def on_prompt(request, body):
        raise httpx.ReadTimeout("timed out", request=request)

    fake = FakeComfy(on_prompt=on_prompt)
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    assert "超时" in str(info.value) and not info.value.retryable
    assert fake.paths("POST") == ["/prompt"]
    assert fake.paths("GET") == ["/history/" + fake.bodies[0]["prompt_id"], "/queue"]  # 只查找，不重新提交


def test_submit_timeout_but_queued_continues():
    def on_prompt(request, body):
        fake.accepted.append(body["prompt_id"])  # 服务端其实已经收下了任务
        raise httpx.ReadTimeout("timed out", request=request)

    fake = FakeComfy(on_prompt=on_prompt, in_queue=True, pending=1)
    result = run({"model": "m"}, fake)
    assert result.image == PNG and fake.paths("POST") == ["/prompt"]
    assert result.params["prompt_id"] == fake.bodies[0]["prompt_id"]


def test_connect_error_is_retried_with_hint():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(DrawError) as info:
        run({"model": "m"}, handler)
    assert calls == ["/prompt", "/prompt"]  # 连接失败说明请求没有发出，可以重试一次
    assert "127.0.0.1" in str(info.value) and "--listen" in str(info.value)


def test_poll_5xx_is_retried():
    fake = FakeComfy()
    flaky = []

    def handler(request):
        if request.url.path.startswith("/history/") and not flaky:
            flaky.append(1)
            return httpx.Response(502, text="Bad Gateway")
        return fake(request)

    assert run({"model": "m"}, handler).image == PNG


def test_execution_error_refusal_message():
    entry = {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {"prompt_id": "x"}],
        ["execution_error", {"node_id": "12", "node_type": "NSFWChecker", "exception_type": "ValueError",
                             "exception_message": "NSFW content detected\n", "traceback": ["..."]}],
    ]}}
    fake = FakeComfy(entry=entry)
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    message = str(info.value)
    assert "节点 #12（NSFWChecker）" in message and "NSFW content detected" in message
    assert fake.paths("POST") == ["/prompt"] and "/view" not in fake.paths()


def test_out_of_memory_and_interrupted_messages():
    oom = {"status": {"status_str": "error", "completed": False, "messages": [["execution_error", {
        "node_id": "3", "node_type": "KSampler", "exception_type": "torch.OutOfMemoryError",
        "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"}]]}}
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(entry=oom))
    assert "减小尺寸" in str(info.value)
    interrupted = {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_interrupted", {"node_id": "3", "node_type": "KSampler"}]]}}
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(entry=interrupted))
    assert "中断" in str(info.value)


def test_no_image_output():
    entry = {"outputs": {"9": {"text": ["hello"]}}, "status": {"status_str": "success", "completed": True}}
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(entry=entry))
    assert "SaveImage" in str(info.value)


def test_view_returns_non_image():
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(view=b"<html>login</html>"))
    assert "不是可识别的图片" in str(info.value)


def test_partial_node_errors_are_reported_as_text():
    def on_prompt(request, body):
        fake.accepted.append(body["prompt_id"])
        return httpx.Response(200, json={"prompt_id": body["prompt_id"], "number": 1, "node_errors": {
            "20": {"errors": [{"message": "Required input is missing", "details": "images"}],
                   "class_type": "PreviewImage"}}})

    fake = FakeComfy(on_prompt=on_prompt)
    result = run({"model": "m"}, fake)
    assert result.image == PNG and "节点 #20 PreviewImage" in result.text


def test_lost_job():
    fake = FakeComfy(pending=10**9, in_queue=False)
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake)
    assert "任务丢失" in str(info.value) and fake.paths("POST") == ["/prompt"]


def test_timeout_cancels_job(monkeypatch):
    monkeypatch.setattr(comfyui, "time", FakeClock())
    fake = FakeComfy(pending=10**9, in_queue=True)
    with pytest.raises(DrawError) as info:
        run({"model": "m", "timeout": 300}, fake)
    pid = fake.bodies[0]["prompt_id"]
    assert "超时" in str(info.value) and "300 秒" in str(info.value) and not info.value.retryable
    assert fake.paths("POST") == ["/prompt", f"/api/jobs/{pid}/cancel"]


def test_timeout_cancel_falls_back_on_old_servers(monkeypatch):
    monkeypatch.setattr(comfyui, "time", FakeClock())
    fake = FakeComfy(pending=10**9, in_queue="pending", cancel_api=False)
    with pytest.raises(DrawError):
        run({"model": "m"}, fake)
    pid = fake.bodies[0]["prompt_id"]
    assert fake.paths("POST") == ["/prompt", f"/api/jobs/{pid}/cancel", "/queue"]
    assert json.loads(fake.requests[-1].content) == {"delete": [pid]}

    monkeypatch.setattr(comfyui, "time", FakeClock())
    fake = FakeComfy(pending=10**9, in_queue=True, cancel_api=False)
    with pytest.raises(DrawError):
        run({"model": "m"}, fake)
    assert fake.paths("POST")[-1] == "/interrupt"
    assert json.loads(fake.requests[-1].content) == {"prompt_id": fake.bodies[0]["prompt_id"]}


def test_cancelled_task_cancels_job():
    fake = FakeComfy(pending=10**9, in_queue=True)

    async def handler(request):
        await REAL_SLEEP(0)
        return fake(request)

    async def go():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            task = asyncio.create_task(ComfyUIBackend({"model": "m"}).generate(Clients(http=pool), DrawRequest("x")))
            while fake.polls < 2:
                await REAL_SLEEP(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await pool.close()

    asyncio.run(go())
    pid = fake.bodies[0]["prompt_id"]
    assert fake.paths("POST") == ["/prompt", f"/api/jobs/{pid}/cancel"]


# ---------------------------------------------------------------- Comfy Cloud


class FakeCloud:
    def __init__(self, *, statuses=("pending", "in_progress", "completed"), job=None, prompt_status=200,
                 prompt_body=None):
        self.statuses = list(statuses)
        self.job = job or {"outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}
        self.prompt_status = prompt_status
        self.prompt_body = prompt_body
        self.upload_status = 200
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.uploads: list[tuple[dict, dict]] = []

    def paths(self, method=None):
        return [r.url.path for r in self.requests if method is None or r.method == method]

    def __call__(self, request):
        self.requests.append(request)
        host, method, path = request.url.host, request.method, request.url.path
        if host == "storage.example.com":
            return httpx.Response(200, content=PNG, headers={"Content-Type": "image/png"})
        assert host == "cloud.comfy.org"
        if method == "POST" and path == "/api/prompt":
            self.bodies.append(json.loads(request.content))
            if self.prompt_status != 200:
                return httpx.Response(self.prompt_status, json=self.prompt_body)
            return httpx.Response(200, json={"prompt_id": "job-1", "number": 0, "node_errors": {}})
        if method == "GET" and path == "/api/jobs/job-1":
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json={"id": "job-1", "status": status, **(
                self.job if status in ("completed", "failed") else {})})
        if method == "GET" and path == "/api/view":
            return httpx.Response(302, headers={"Location": "https://storage.example.com/out.png?sig=abc"})
        if method == "GET" and path == "/api/experiment/models/checkpoints":
            return httpx.Response(200, json=[{"name": "sdxl_cloud.safetensors", "pathIndex": 0},
                                             {"name": "flux.safetensors", "pathIndex": 0}])
        if method == "POST" and path in ("/api/queue", "/api/interrupt"):
            return httpx.Response(200, json={})
        if method == "POST" and path == "/api/upload/image":
            self.uploads.append(parse_multipart(request))
            if self.upload_status != 200:
                return httpx.Response(self.upload_status, json={"message": "User has not paid"})
            # Cloud 按内容哈希存储，返回的 name 与上传的文件名无关
            return httpx.Response(200, json={"name": "0f1e2d3c4b5a.jpg", "subfolder": "", "type": "input"})
        if method == "GET" and path == "/api/object_info":
            return httpx.Response(200, json=OBJECT_INFO)
        return httpx.Response(404, json={"code": "not_found", "message": "Not found"})


CLOUD = {"base_url": "https://cloud.comfy.org/api/", "api_key": "comfyui-key"}


def test_cloud_flow():
    fake = FakeCloud()
    result = run(CLOUD, fake, DrawRequest("1girl", "lowres", "square", seed=3))
    assert fake.paths() == ["/api/experiment/models/checkpoints", "/api/prompt", "/api/jobs/job-1",
                            "/api/jobs/job-1", "/api/jobs/job-1", "/api/view", "/out.png"]
    body = fake.bodies[0]
    # Cloud 的请求体没有 prompt_id / client_id；付费 API 节点需要的 comfy.org Key 与 X-API-Key 相同，自动带上
    assert set(body) == {"prompt", "extra_data"} and body["extra_data"] == {"api_key_comfy_org": "comfyui-key"}
    assert body["prompt"]["4"]["inputs"]["ckpt_name"] == "sdxl_cloud.safetensors"
    assert body["prompt"]["5"]["inputs"]["width"] == 1024
    for request in fake.requests:
        if request.url.host == "cloud.comfy.org":
            assert request.headers["x-api-key"] == "comfyui-key" and "authorization" not in request.headers
        else:  # 签名下载地址不能带 Key
            assert "x-api-key" not in request.headers and "authorization" not in request.headers
    assert result.image == PNG and result.params["prompt_id"] == "job-1" and result.params["size"] == "1024x1024"


def test_cloud_list_models():
    assert run(CLOUD, FakeCloud(), action="models") == ["sdxl_cloud.safetensors", "flux.safetensors"]


def test_cloud_partner_node_key_is_added_without_overriding_config():
    # 官方文档：直接调用 Cloud API 时，付费 API 节点要在 extra_data.api_key_comfy_org 中带上 Key（与 X-API-Key 相同）
    extra = {"extra_data": {"api_key_comfy_org": "comfyui-other", "x": 1}, "front": True}
    section = {**CLOUD, "model": "m", "extra_body": extra}
    fake = FakeCloud()
    run(section, fake, DrawRequest("1girl"))
    assert fake.bodies[0]["extra_data"] == {"api_key_comfy_org": "comfyui-other", "x": 1}  # 配置中的值优先
    fake = FakeCloud()
    run({**CLOUD, "model": "m", "extra_body": {"extra_data": {"x": 1}}}, fake, edit_req(jpeg()))
    assert fake.bodies[0]["extra_data"] == {"api_key_comfy_org": "comfyui-key", "x": 1}
    assert extra == {"extra_data": {"api_key_comfy_org": "comfyui-other", "x": 1}, "front": True}  # 不修改配置
    # 本地 ComfyUI 的 api_key 是反向代理的 Key，不能当作 comfy.org Key 放进请求体
    fake = FakeComfy()
    run({"model": "m", "api_key": "proxy-secret"}, fake)
    prompt = next(r for r in fake.requests if r.url.path == "/prompt")
    assert "extra_data" not in fake.bodies[0] and b"proxy-secret" not in prompt.content


@pytest.mark.parametrize(("status", "expected"), [(402, "余额不足"), (429, "订阅未激活")])
def test_cloud_billing_errors_are_not_retried(status, expected):
    fake = FakeCloud(prompt_status=status, prompt_body={"message": "User has not paid"})
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake)
    assert info.value.status == status and not info.value.retryable and expected in str(info.value)
    assert fake.paths("POST") == ["/api/prompt"]


def test_cloud_validation_and_auth_errors():
    fake = FakeCloud(prompt_status=400, prompt_body={"error": {"type": "invalid_prompt", "message": "Bad graph"},
                                                     "node_errors": {}})
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake)
    assert "Bad graph" in str(info.value) and "Comfy Cloud" in str(info.value)
    fake = FakeCloud(prompt_status=401, prompt_body={"code": "unauthorized", "message": "Invalid API key"})
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake)
    assert info.value.status == 401 and "Invalid API key" in str(info.value) and fake.paths("POST") == ["/api/prompt"]


def test_cloud_job_failure():
    fake = FakeCloud(statuses=("failed",), job={"execution_error": {
        "node_id": "3", "node_type": "KSampler", "exception_type": "RuntimeError", "exception_message": "boom"}})
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake)
    assert "Comfy Cloud 执行失败" in str(info.value) and "boom" in str(info.value)


def test_cloud_job_not_found_right_after_submit_is_tolerated():
    fake = FakeCloud()
    first = []

    def handler(request):
        if request.url.path == "/api/jobs/job-1" and not first:
            first.append(1)
            return httpx.Response(404, json={"code": "not_found", "message": "Job not found"})
        return fake(request)

    assert run({**CLOUD, "model": "m"}, handler).image == PNG


def test_cloud_cancelled_task_dequeues_pending_job():
    fake = FakeCloud(statuses=("pending",))

    async def handler(request):
        await REAL_SLEEP(0)
        return fake(request)

    async def go():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            task = asyncio.create_task(ComfyUIBackend({**CLOUD, "model": "m"}).generate(Clients(http=pool),
                                                                                         DrawRequest("x")))
            while fake.paths("GET").count("/api/jobs/job-1") < 2:
                await REAL_SLEEP(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await pool.close()

    asyncio.run(go())
    # 排队中的任务只从队列删除；/api/interrupt 会中断该账号所有运行中的任务，不调用
    assert fake.paths("POST") == ["/api/prompt", "/api/queue"]
    assert json.loads(fake.requests[-1].content) == {"delete": ["job-1"]}


def test_cloud_timeout_dequeues_and_interrupts(monkeypatch):
    monkeypatch.setattr(comfyui, "time", FakeClock())
    fake = FakeCloud(statuses=("in_progress",))
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake)
    assert "超时" in str(info.value)
    assert fake.paths("POST") == ["/api/prompt", "/api/queue", "/api/interrupt"]
    queue = [r for r in fake.requests if r.url.path == "/api/queue"][0]
    assert json.loads(queue.content) == {"delete": ["job-1"]}


# ---------------------------------------------------------------- 图片修改


def jpeg(width=1200, height=800, data=JPG) -> InputImage:
    return InputImage(data, "image/jpeg", width, height)


def edit_req(*images, orientation="landscape", strength=0.45, prompt="1girl, night sky", negative="lowres"):
    return DrawRequest(prompt, negative, orientation, seed=42, images=list(images), strength=strength)


def upload_name(data: bytes, ext: str) -> str:
    return f"multidraw_{hashlib.sha256(data).hexdigest()[:32]}{ext}"


def test_edit_builtin_img2img_request_shape():
    fake = FakeComfy()
    result = run({}, fake, edit_req(jpeg(1200, 800)))
    body = fake.bodies[0]
    pid = body["prompt_id"]
    # 先确定模型（不发多余的上传），再上传图片、提交工作流、轮询、下载
    assert fake.paths() == ["/models/checkpoints", "/upload/image", "/prompt", f"/history/{pid}", f"/history/{pid}",
                            "/view"]
    upload = next(r for r in fake.requests if r.url.path == "/upload/image")
    assert upload.headers["content-type"].startswith("multipart/form-data; boundary=")
    name = upload_name(JPG, ".jpg")
    assert fake.uploads == [({"type": "input", "overwrite": "false", "subfolder": "astrbot"},
                             {"image": (name, "image/jpeg", JPG)})]
    graph = body["prompt"]
    assert set(graph) == {"3", "4", "6", "7", "8", "9", "10", "11", "12"}  # 没有 EmptyLatentImage
    assert graph["10"] == {"class_type": "LoadImage", "inputs": {"image": f"astrbot/{name}"}}
    assert graph["11"] == {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["10", 0], "upscale_method": "lanczos", "megapixels": 0.96, "resolution_steps": 8}}
    assert graph["12"]["inputs"] == {"pixels": ["11", 0], "vae": ["4", 2]}
    sampler = graph["3"]["inputs"]
    assert (sampler["denoise"], sampler["latent_image"], sampler["seed"]) == (0.45, ["12", 0], 42)
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == \
        (25, 7.0, "dpmpp_2m", "karras")
    assert graph["4"]["inputs"]["ckpt_name"] == CHECKPOINTS[0]
    assert graph["6"]["inputs"]["text"] == "1girl, night sky" and graph["7"]["inputs"]["text"] == "lowres"
    assert graph["9"]["inputs"]["filename_prefix"] == "astrbot/edit"
    assert all("authorization" not in r.headers for r in fake.requests)
    assert result.image == PNG and result.text == ""
    # 1200x800 按 0.96 百万像素、对齐到 8 缩放后为 1232x816（与 ImageScaleToTotalPixels 一致，保持原图比例）
    assert result.params == {
        "mode": "edit", "model": CHECKPOINTS[0], "size": "1232x816", "seed": 42, "input_images": 1,
        "workflow": "builtin-img2img", "prompt_id": pid, "strength": 0.45, "steps": 25, "cfg": 7.0,
        "sampler": "dpmpp_2m", "scheduler": "karras",
    }


@pytest.mark.parametrize(("strength", "expected"), [(0.0, 0.6), (0.2, 0.2), (0.456, 0.46), (1.5, 0.6),
                                                     (-1, 0.6), (0.004, 0.01), ("abc", 0.6)])
def test_edit_strength_default_and_range(strength, expected):
    fake = FakeComfy()
    result = run({"model": "m"}, fake, edit_req(jpeg(), strength=strength))
    assert graph_of(fake)["3"]["inputs"]["denoise"] == expected and result.params["strength"] == expected


def test_edit_keeps_aspect_or_crops_to_requested_orientation():
    # 方向与原图一致（main.py 默认按原图推断）：保持原图比例，总像素按该方向的尺寸设置
    fake = FakeComfy()
    result = run({"model": "m", "square_size": "1024x1024"}, fake, edit_req(jpeg(1000, 1000), orientation="square"))
    assert graph_of(fake)["11"]["inputs"]["megapixels"] == 1.0 and result.params["size"] == "1024x1024"

    # 明确要求了与原图不同的方向：居中裁切到该方向的尺寸
    fake = FakeComfy()
    result = run({"model": "m"}, fake, edit_req(jpeg(1200, 800), orientation="portrait"))
    assert graph_of(fake)["11"] == {"class_type": "ImageScale", "inputs": {
        "image": ["10", 0], "upscale_method": "lanczos", "width": 832, "height": 1216, "crop": "center"}}
    assert graph_of(fake)["12"]["inputs"]["pixels"] == ["11", 0] and result.params["size"] == "832x1216"

    # 不知道原图尺寸：仍按总像素缩放（保持原图比例）
    fake = FakeComfy()
    result = run({"model": "m"}, fake, edit_req(jpeg(0, 0), orientation="portrait"))
    assert graph_of(fake)["11"]["class_type"] == "ImageScaleToTotalPixels" and result.params["size"] == "0.96MP"


def test_edit_transparent_png_is_flattened_on_white():
    fake = FakeComfy()
    png = InputImage(PNG, "image/png", 512, 768)
    result = run({"model": "m"}, fake, edit_req(png, orientation="portrait"))
    graph = graph_of(fake)
    assert fake.uploads[0][1]["image"] == (upload_name(PNG, ".png"), "image/png", PNG)
    comp_id = next(k for k, node in graph.items() if node["class_type"] == "ImageCompositeMasked")
    comp = graph[comp_id]["inputs"]
    white, invert = comp["destination"][0], comp["mask"][0]
    assert graph[white] == {"class_type": "EmptyImage", "inputs": {
        "width": 512, "height": 768, "batch_size": 1, "color": 0xFFFFFF}}
    assert graph[invert] == {"class_type": "InvertMask", "inputs": {"mask": ["10", 1]}}
    assert comp["source"] == ["10", 0] and comp["resize_source"] is True
    assert graph["11"]["inputs"]["image"] == [comp_id, 0]  # 缩放节点改为读取合成后的图片
    assert result.params["alpha"] == "white" and result.params["size"] == "816x1232"
    # JPEG（没有透明通道）不插入合成节点
    fake = FakeComfy()
    result = run({"model": "m"}, fake, edit_req(jpeg()))
    assert "ImageCompositeMasked" not in json.dumps(graph_of(fake)) and "alpha" not in result.params


@pytest.mark.parametrize("workflow", ["{}", "  ", json.dumps({"prompt": comfyui.DEFAULT_EDIT_WORKFLOW})],
                         ids=["empty-object", "blank", "pasted-builtin"])
def test_edit_empty_or_builtin_workflow_config_is_treated_as_builtin(workflow):
    # 修改工作流填 "{}" 或粘贴了内置工作流时实际运行的是内置工作流：同样铺白底、换方向时裁切，记录为 builtin
    fake = FakeComfy()
    png = InputImage(PNG, "image/png", 1200, 800)
    result = run({"model": "m", "edit_workflow_json": workflow}, fake, edit_req(png, orientation="portrait"))
    graph = graph_of(fake)
    assert result.params["workflow"] == "builtin-img2img" and result.params["alpha"] == "white"
    assert graph["11"]["class_type"] == "ImageScale" and result.params["size"] == "832x1216"
    assert "ImageCompositeMasked" in json.dumps(graph)


def test_edit_extra_images_are_trimmed_with_note():
    fake = FakeComfy()
    images = [jpeg(), jpeg(data=JPG + b"2"), jpeg(data=JPG + b"3")]
    result = run({"model": "m"}, fake, edit_req(*images))
    assert len(fake.uploads) == 1 and fake.uploads[0][1]["image"][2] == JPG  # 内置工作流只用第 1 张
    assert result.params["input_images"] == 1 and result.params["ignored_images"] == 2
    assert "只使用前 1 张图片，其余 2 张已忽略" in result.text


QWEN_EDIT = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "{{image1}}"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "{{image2}}"}},
    "3": {"class_type": "LoadImage", "inputs": {"image": "{{image3}}"}},
    "31": {"class_type": "ImageScale", "inputs": {"image": ["3", 0], "upscale_method": "area", "width": 512,
                                                   "height": 512, "crop": "disabled"}},
    "20": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.safetensors", "type": "qwen_image"}},
    "21": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    "22": {"class_type": "UNETLoader", "inputs": {"unet_name": "qie.safetensors", "weight_dtype": "default"}},
    "4": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
        "clip": ["20", 0], "vae": ["21", 0], "prompt": "{{prompt}}",
        "image1": ["1", 0], "image2": ["2", 0], "image3": ["31", 0]}},
    "5": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
        "clip": ["20", 0], "vae": ["21", 0], "prompt": "", "image1": ["1", 0]}},
    "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "7": {"class_type": "KSampler", "inputs": {"model": ["22", 0], "positive": ["4", 0], "negative": ["5", 0],
                                               "latent_image": ["6", 0], "seed": 1, "steps": 8, "cfg": 1.0,
                                               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["21", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "qie"}},
}


def test_edit_custom_multi_image_workflow_prunes_unused_slots():
    section = {"edit_workflow_json": json.dumps(QWEN_EDIT), "model": "unused.safetensors"}
    backend = ComfyUIBackend(section)
    assert backend.max_edit_images == 3 and backend.supports_strength is False and backend.edit_missing() == ""
    fake = FakeComfy()
    second = InputImage(PNG, "image/png", 0, 0)
    result = run(section, fake, edit_req(jpeg(1200, 800), second, prompt="put the cat from image 2 on the sofa"))
    graph = graph_of(fake)
    # 只发了 2 张图：第 3 个加载节点和只为它服务的缩放节点被删除，编码节点只丢了可选的 image3，保留
    assert "3" not in graph and "31" not in graph
    assert set(graph["4"]["inputs"]) == {"clip", "vae", "prompt", "image1", "image2"}
    names = [files["image"][0] for _, files in fake.uploads]
    assert names == [upload_name(JPG, ".jpg"), upload_name(PNG, ".png")]
    assert graph["1"]["inputs"]["image"] == f"astrbot/{names[0]}" and graph["2"]["inputs"]["image"] == f"astrbot/{names[1]}"
    assert graph["4"]["inputs"]["prompt"] == "put the cat from image 2 on the sofa"
    # 固定种子被替换；没有 {{width}} / {{height}} 时按第 1 张图的比例设置 Empty Latent（对齐到 64）
    assert graph["7"]["inputs"]["seed"] == 42 and graph["7"]["inputs"]["denoise"] == 1.0
    assert (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]) == (1216, 832)
    assert "ImageCompositeMasked" not in json.dumps(graph)  # 自定义工作流不插入白底合成节点
    assert "/models/checkpoints" not in fake.paths() and fake.object_info_calls > 0
    assert result.params == {"mode": "edit", "model": "", "size": "1216x832", "seed": 42, "input_images": 2,
                             "workflow": "custom-edit", "prompt_id": fake.bodies[0]["prompt_id"]}

    # /object_info 的结果有缓存：再次修改不重复查询
    fake = FakeComfy()
    run(section, fake, edit_req(jpeg()))
    assert fake.object_info_calls == 0 and "2" not in graph_of(fake) and len(fake.uploads) == 1


FLUX2_EDIT = {
    "L1": {"class_type": "LoadImage", "inputs": {"image": "{{image1}}"}},
    "L2": {"class_type": "LoadImage", "inputs": {"image": "{{image2}}"}},
    "S1": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["L1", 0], "upscale_method": "lanczos",
                                                               "megapixels": 1.0, "resolution_steps": 1}},
    "S2": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["L2", 0], "upscale_method": "lanczos",
                                                               "megapixels": 1.0, "resolution_steps": 1}},
    "E1": {"class_type": "VAEEncode", "inputs": {"pixels": ["S1", 0], "vae": ["V", 0]}},
    "E2": {"class_type": "VAEEncode", "inputs": {"pixels": ["S2", 0], "vae": ["V", 0]}},
    "V": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
    "C": {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral.safetensors", "type": "flux2"}},
    "T": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["C", 0]}},
    "R1": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["T", 0], "latent": ["E1", 0]}},
    "R2": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["R1", 0], "latent": ["E2", 0]}},
    "U": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2.safetensors", "weight_dtype": "default"}},
    "K": {"class_type": "KSampler", "inputs": {"model": ["U", 0], "positive": ["R2", 0], "negative": ["T", 0],
                                               "latent_image": ["EL", 0], "seed": 1, "steps": 20, "cfg": 1.0,
                                               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "EL": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "D": {"class_type": "VAEDecode", "inputs": {"samples": ["K", 0], "vae": ["V", 0]}},
    "O": {"class_type": "SaveImage", "inputs": {"images": ["D", 0], "filename_prefix": "x"}},
}


def test_edit_pruning_follows_required_inputs():
    section = {"edit_workflow_json": json.dumps(FLUX2_EDIT)}
    fake = FakeComfy()
    run(section, fake, edit_req(jpeg()))
    graph = graph_of(fake)
    # VAEEncode 丢了必填的 pixels → 一起删除；ReferenceLatent 只丢了可选的 latent → 保留
    assert not {"L2", "S2", "E2"} & set(graph)
    assert graph["R2"]["inputs"] == {"conditioning": ["R1", 0]} and graph["K"]["inputs"]["positive"] == ["R2", 0]
    assert any(r.url.path == "/object_info/VAEEncode" for r in fake.requests)

    # 查不到 /object_info（如旧版或反向代理拦截）时退回「没有连线就删」
    comfyui._REQUIRED_CACHE.clear()
    fake = FakeComfy(object_info_status=500)
    run(section, fake, edit_req(jpeg()))
    graph = graph_of(fake)
    assert "S2" not in graph and graph["E2"]["inputs"] == {"vae": ["V", 0]}

    # 第 2 张图连到了决定输出尺寸的必填输入：删到连 SaveImage 都没有时直接说明，不上传、不提交
    needs_two = json.loads(json.dumps(FLUX2_EDIT))
    needs_two["GS"] = {"class_type": "GetImageSize", "inputs": {"image": ["L2", 0]}}
    needs_two["EL"]["inputs"].update(width=["GS", 0], height=["GS", 1])
    comfyui._REQUIRED_CACHE.clear()
    fake = FakeComfy()
    with pytest.raises(DrawError) as info:
        run({"edit_workflow_json": json.dumps(needs_two)}, fake, edit_req(jpeg()))
    assert "只有 1 张图片时无法运行" in str(info.value) and "SaveImage" in str(info.value)
    assert not fake.uploads and "/prompt" not in fake.paths()


def test_edit_workflow_without_placeholders_fills_load_image_nodes_in_order():
    workflow = json.loads(json.dumps(QWEN_EDIT).replace('"{{image1}}"', '"a.png"').replace('"{{image2}}"', '"b.png"'))
    for key in ("3", "31"):
        workflow.pop(key)
    workflow["4"]["inputs"].pop("image3")
    section = {"edit_workflow_json": workflow}  # 也接受已经是对象的配置
    assert ComfyUIBackend(section).max_edit_images == 2
    fake = FakeComfy()
    run(section, fake, edit_req(jpeg()))
    graph = graph_of(fake)
    assert graph["1"]["inputs"]["image"].startswith("astrbot/multidraw_") and "2" not in graph
    assert "image2" not in graph["4"]["inputs"] and workflow["1"]["inputs"]["image"] == "a.png"  # 不修改配置

    # 超过 4 个 LoadImage 节点时最多使用前 4 个，其余保持原样（如固定的参考图）
    many = {str(i): {"class_type": "LoadImage", "inputs": {"image": f"ref{i}.png"}} for i in range(1, 7)}
    many["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["1", 0]}}
    assert ComfyUIBackend({"edit_workflow_json": json.dumps(many)}).max_edit_images == 4


def test_edit_capabilities_and_edit_missing():
    # 类属性是平台的最大能力，实例按当前配置计算
    assert ComfyUIBackend.max_edit_images == comfyui.MAX_EDIT_IMAGES == 4
    assert ComfyUIBackend.supports_strength is True and ComfyUIBackend.supports_balance is True
    assert ComfyUIBackend.billing_url == comfyui.BILLING_URL and ComfyUIBackend.billing_url.startswith("https://")
    default = ComfyUIBackend({})
    assert default.max_edit_images == 1 and default.supports_strength is True and default.edit_missing() == ""
    assert ComfyUIBackend(None).edit_missing() == ""
    with_denoise = json.loads(json.dumps(QWEN_EDIT))
    with_denoise["7"]["inputs"]["denoise"] = "{{denoise}}"
    assert ComfyUIBackend({"edit_workflow_json": json.dumps(with_denoise)}).supports_strength is True
    # 文生图工作流有误不影响修改图片
    broken = ComfyUIBackend({"workflow_json": "{oops"})
    assert broken.missing() and broken.edit_missing() == ""
    # 没有 {{prompt}} 但采样器 positive 直接连着 CLIPTextEncode：生成时会自动填入，不算配置有误
    wired = json.loads(json.dumps(comfyui.DEFAULT_EDIT_WORKFLOW))
    wired["6"]["inputs"]["text"] = "a photo"
    assert ComfyUIBackend({"edit_workflow_json": json.dumps(wired)}).edit_missing() == ""

    cases = [
        ({"edit_workflow_json": "{oops"}, "不是有效的 JSON"),
        ({"edit_workflow_json": '{"nodes": [], "links": []}'}, "Save (API Format)"),
        ({"edit_workflow_json": json.dumps(FLUX_WORKFLOW)}, "没有 {{image}} 占位符"),
        ({"edit_workflow_json": '{"1": {"class_type": "LoadImage", "inputs": {"image": "{{image5}}"}}}'},
         "最多使用 4 张图片"),
        ({"edit_workflow_json": '{"1": {"class_type": "LoadImage", "inputs": {"image": "{{mask}}"}}}'},
         "未知的占位符 {{mask}}"),
        # Qwen-Image-Edit 的提示词输入叫 prompt，不能自动连线：没写 {{prompt}} 时生成必然失败，提前说明、不上传
        ({"edit_workflow_json": json.dumps(QWEN_EDIT).replace('"{{prompt}}"', '"make it night"')},
         "没有 {{prompt}} 占位符"),
        ({"extra_body": "{bad"}, "额外请求参数"),
        ({"base_url": "https://cloud.comfy.org"}, "Comfy Cloud API Key"),
        ({"base_url": "http://[bad"}, "地址格式不正确"),
    ]
    for section, expected in cases:
        backend = ComfyUIBackend(section)
        assert expected in backend.edit_missing(), section
        assert 1 <= backend.max_edit_images <= 4 and isinstance(backend.supports_strength, bool)
        fake = FakeComfy()
        with pytest.raises(DrawError) as info:
            run(section, fake, edit_req(jpeg()))
        assert expected in str(info.value) and fake.requests == [], section


def test_text_workflow_rejects_edit_placeholders():
    workflow = json.loads(json.dumps(comfyui.DEFAULT_WORKFLOW))
    workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}}
    fake = FakeComfy()
    with pytest.raises(DrawError) as info:
        run({"workflow_json": json.dumps(workflow)}, fake)
    assert "未知的占位符 {{image}}" in str(info.value) and "图片修改工作流" in str(info.value)
    assert fake.requests == []
    # 图片修改工作流里写在文字中的占位符按文字替换
    edit = json.loads(json.dumps(comfyui.DEFAULT_EDIT_WORKFLOW))
    edit["9"]["inputs"]["filename_prefix"] = "edit {{denoise}} {{megapixels}}"
    fake = FakeComfy()
    run({"model": "m", "edit_workflow_json": json.dumps(edit)}, fake, edit_req(jpeg(), strength=0.5))
    assert graph_of(fake)["9"]["inputs"]["filename_prefix"] == "edit 0.5 0.96"


def test_edit_upload_retries_and_errors():
    # 上传按内容哈希命名、overwrite=false，5xx / 超时后重试不会产生重复文件
    fake = FakeComfy(upload_fail=[500, 502])
    assert run({"model": "m"}, fake, edit_req(jpeg())).image == PNG
    assert len(fake.uploads) == 3 and len({files["image"][0] for _, files in fake.uploads}) == 1
    assert fake.paths("POST").count("/prompt") == 1

    calls = []
    fake = FakeComfy()

    def timeout_once(request):
        if request.url.path == "/upload/image" and not calls:
            calls.append(1)
            raise httpx.ReadTimeout("timed out", request=request)
        return fake(request)

    assert run({"model": "m"}, timeout_once, edit_req(jpeg())).image == PNG and len(fake.uploads) == 1

    fake = FakeComfy(upload_fail=[413])
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake, edit_req(jpeg()))
    assert info.value.status == 413 and "--max-upload-size" in str(info.value)
    assert len(fake.uploads) == 1 and "/prompt" not in fake.paths()

    fake = FakeComfy(upload_fail=[401])
    with pytest.raises(DrawError) as info:
        run({"model": "m", "api_key": "proxy-secret"}, fake, edit_req(jpeg()))
    assert info.value.status == 401 and len(fake.uploads) == 1 and "proxy-secret" not in str(info.value)

    fake = FakeComfy(upload_fail=[500, 500, 500])
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake, edit_req(jpeg()))
    assert info.value.status == 500 and len(fake.uploads) == comfyui.UPLOAD_ATTEMPTS and "/prompt" not in fake.paths()


def test_edit_submit_errors_are_not_retried():
    fake = FakeComfy(on_prompt=lambda request, body: httpx.Response(500, text="500 Internal Server Error"))
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake, edit_req(jpeg()))
    assert info.value.status == 500 and fake.paths("POST") == ["/upload/image", "/prompt"]

    def invalid_image(request, body):
        return httpx.Response(400, json={
            "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                      "details": ""},
            "node_errors": {"10": {"errors": [{"type": "custom_validation_failed",
                                               "message": "Custom validation failed for node",
                                               "details": "image - Invalid image file: astrbot/x.jpg",
                                               "extra_info": {"input_name": "image"}}],
                                   "class_type": "LoadImage"}}})

    fake = FakeComfy(on_prompt=invalid_image)
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake, edit_req(jpeg()))
    assert "节点 #10 LoadImage.image" in str(info.value) and "Invalid image file" in str(info.value)
    assert not info.value.retryable and fake.paths("POST") == ["/upload/image", "/prompt"]


def test_edit_timeout_cancels_job(monkeypatch):
    monkeypatch.setattr(comfyui, "time", FakeClock())
    fake = FakeComfy(pending=10**9, in_queue=True)
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, fake, edit_req(jpeg()))
    pid = fake.bodies[0]["prompt_id"]
    assert "超时" in str(info.value) and not info.value.retryable
    assert fake.paths("POST") == ["/upload/image", "/prompt", f"/api/jobs/{pid}/cancel"]


def test_moderation_refusal_gets_friendly_message():
    entry = {"status": {"status_str": "error", "completed": False, "messages": [["execution_error", {
        "node_id": "5", "node_type": "OpenAIGPTImage1", "exception_type": "Exception",
        "exception_message": "Your request was rejected as a result of our safety system."}]]}}
    with pytest.raises(DrawError) as info:
        run({"model": "m"}, FakeComfy(entry=entry), edit_req(jpeg()))
    assert "安全审核" in str(info.value) and "safety system" in str(info.value)


def test_cloud_edit_flow():
    fake = FakeCloud()
    result = run({**CLOUD, "model": "sdxl.safetensors"}, fake, edit_req(jpeg()))
    assert fake.paths() == ["/api/upload/image", "/api/prompt", "/api/jobs/job-1", "/api/jobs/job-1",
                            "/api/jobs/job-1", "/api/view", "/out.png"]
    upload = fake.requests[0]
    assert str(upload.url) == "https://cloud.comfy.org/api/upload/image"
    assert upload.headers["x-api-key"] == "comfyui-key" and "authorization" not in upload.headers
    # Cloud 不写 type 时默认 output；subfolder 被忽略，不发送
    assert fake.uploads == [({"type": "input", "overwrite": "false"},
                             {"image": (upload_name(JPG, ".jpg"), "image/jpeg", JPG)})]
    body = fake.bodies[0]
    assert set(body) == {"prompt", "extra_data"} and body["prompt"]["10"]["inputs"]["image"] == "0f1e2d3c4b5a.jpg"
    assert result.params["mode"] == "edit" and result.params["prompt_id"] == "job-1"

    # Cloud 只提供整表 /api/object_info
    fake = FakeCloud()
    run({**CLOUD, "edit_workflow_json": json.dumps(FLUX2_EDIT)}, fake, edit_req(jpeg()))
    assert fake.paths("GET").count("/api/object_info") == 1 and "E2" not in fake.bodies[0]["prompt"]

    # 订阅未激活（429）不重试
    fake = FakeCloud()
    fake.upload_status = 429
    with pytest.raises(DrawError) as info:
        run({**CLOUD, "model": "m"}, fake, edit_req(jpeg()))
    assert info.value.status == 429 and "订阅未激活" in str(info.value) and len(fake.uploads) == 1
    assert "/api/prompt" not in fake.paths()


# ---------------------------------------------------------------- 余额


TIMESERIES = {"group_by": "model", "granularity": "month", "groups": [], "buckets": [], "breakdown": [],
              "summary": {"spend_micros": 3210000, "balance": {
                  "amount_micros": 4250, "prepaid_balance_micros": 2250, "cloud_credit_balance_micros": 2000,
                  "currency": "USD"}}}
PLATFORM_BALANCE = {"amount_micros": 1234.25, "amount_cents": 1234.25, "currency": "usd",
                    "prepaid_balance_cents": 1000, "cloud_credit_balance_cents": 234.25}


class FakeBilling:
    def __init__(self, *, timeseries=TIMESERIES, timeseries_status=200, platform=PLATFORM_BALANCE,
                 platform_status=200):
        self.timeseries, self.timeseries_status = timeseries, timeseries_status
        self.platform, self.platform_status = platform, platform_status
        self.requests: list[httpx.Request] = []

    def urls(self):
        return [f"{r.method} {r.url.host}{r.url.path}" for r in self.requests]

    def __call__(self, request):
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "cloud.comfy.org" and path == "/api/billing/usage/timeseries":
            if self.timeseries_status != 200:
                return httpx.Response(self.timeseries_status, json={"code": "ERR", "message": "billing failed"})
            return httpx.Response(200, json=self.timeseries)
        if host == "api.comfy.org" and path == "/customers/balance":
            if self.platform_status != 200:
                return httpx.Response(self.platform_status, json={"error": "e", "message": "Customer not found"
                                                                  if self.platform_status == 404 else "Invalid Comfy API key"})
            return httpx.Response(200, json=self.platform)
        return httpx.Response(404, json={"message": "Not found"})


PLATFORM_ITEMS = [("剩余额度", "$12.34（约 2,604 积分）"), ("订阅赠送额度", "$2.34"), ("充值额度", "$10.00")]


def test_cloud_balance_from_usage_timeseries():
    fake = FakeBilling()
    info = run(CLOUD, fake, action="balance")
    assert info.items == [("剩余额度", "$42.50（约 8,968 积分）"), ("订阅赠送额度", "$20.00"), ("充值额度", "$22.50"),
                          ("近一个月消费", "$3.21")]
    assert "账单" in info.note
    assert fake.urls() == ["GET cloud.comfy.org/api/billing/usage/timeseries"]
    request = fake.requests[0]
    assert dict(request.url.params) == {"granularity": "month", "months": "1"}
    assert request.headers["x-api-key"] == "comfyui-key" and "authorization" not in request.headers
    assert "comfyui-key" not in json.dumps(info.items, ensure_ascii=False) + info.note


@pytest.mark.parametrize(("timeseries", "status", "timeseries_calls"), [
    ({"summary": {"spend_micros": 0}}, 200, 1),  # 读不到余额时省略 summary.balance
    ({"groups": []}, 200, 1),
    (TIMESERIES, 500, 2),  # 5xx 重试一次后改用 api.comfy.org
    (TIMESERIES, 404, 1),
])
def test_cloud_balance_falls_back_to_platform(timeseries, status, timeseries_calls):
    fake = FakeBilling(timeseries=timeseries, timeseries_status=status)
    info = run(CLOUD, fake, action="balance")
    assert info.items == PLATFORM_ITEMS and "团队" in info.note
    assert fake.urls() == ["GET cloud.comfy.org/api/billing/usage/timeseries"] * timeseries_calls + \
        ["GET api.comfy.org/customers/balance"]
    assert fake.requests[-1].headers["x-api-key"] == "comfyui-key"


@pytest.mark.parametrize("status", [400, 401, 402, 429])
def test_cloud_balance_key_errors_are_raised(status):
    fake = FakeBilling(timeseries_status=status)
    with pytest.raises(DrawError) as info:
        run(CLOUD, fake, action="balance")
    assert info.value.status == status and "comfyui-key" not in str(info.value)
    assert fake.urls() == ["GET cloud.comfy.org/api/billing/usage/timeseries"]  # 不重试，也不换接口


def test_local_balance_uses_comfy_org_key_from_extra_body():
    section = {"api_key": "proxy-secret",
               "extra_body": json.dumps({"extra_data": {"api_key_comfy_org": "comfyui-local"}})}
    assert ComfyUIBackend(section).balance_missing() == ""
    fake = FakeBilling()
    info = run(section, fake, action="balance")
    assert info.items == PLATFORM_ITEMS and "共用" in info.note and "团队" not in info.note
    assert fake.urls() == ["GET api.comfy.org/customers/balance"]
    # 只发送 comfy.org 的 Key，反向代理的 Key 不会发给 api.comfy.org
    assert fake.requests[0].headers["x-api-key"] == "comfyui-local" and "authorization" not in fake.requests[0].headers

    fake = FakeBilling(platform_status=404)
    with pytest.raises(DrawError) as info:
        run(section, fake, action="balance")
    assert info.value.status == 404 and "还没有积分账户" in str(info.value)

    fake = FakeBilling(platform_status=401)
    with pytest.raises(DrawError) as info:
        run(section, fake, action="balance")
    message = str(info.value)
    assert info.value.status == 401 and "extra_data.api_key_comfy_org" in message
    assert "comfyui-local" not in message and "proxy-secret" not in message


def test_balance_missing_cases():
    local = ComfyUIBackend({})
    assert "不计费" in local.balance_missing() and local.supports_balance is True
    assert "不计费" in ComfyUIBackend({"extra_body": "{bad"}).balance_missing()
    assert "不计费" in ComfyUIBackend({"extra_body": '{"extra_data": {"api_key_comfy_org": ""}}'}).balance_missing()
    assert "Comfy Cloud API Key" in ComfyUIBackend({"base_url": "https://cloud.comfy.org"}).balance_missing()
    assert ComfyUIBackend(CLOUD).balance_missing() == ""
    fake = FakeBilling()
    with pytest.raises(DrawError) as info:
        run({}, fake, action="balance")
    assert "不计费" in str(info.value) and fake.requests == []


def test_balance_items_formatting():
    assert comfyui.balance_items({"amount_cents": -123, "currency": "usd"}) == [("剩余额度", "-$1.23（约 -260 积分）")]
    assert comfyui.balance_items({"amount_micros": 500, "currency": "eur"}) == [("剩余额度", "5.00 EUR")]
    # *_cents 优先于已废弃的 *_micros；可选字段按需显示
    items = comfyui.balance_items({"amount_cents": 100, "amount_micros": 999, "pending_charges_cents": 25,
                                   "effective_balance_cents": 75, "prepaid_balance_micros": True})
    assert items == [("剩余额度", "$1.00（约 211 积分）"), ("待结算费用", "$0.25"), ("扣除待结算后", "$0.75")]
    with pytest.raises(DrawError):
        comfyui.balance_items({"currency": "usd"})


# ---------------------------------------------------------------- 配置 schema


def schema_items() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["comfyui"]["items"]


def test_schema_section_matches_code():
    items = schema_items()
    keys = list(items)
    assert keys[:7] == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                        "square_size"]
    assert keys[-2:] == ["timeout", "extra_body"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field["description"] and field["hint"], name
        if "options" in field:
            assert field["default"] in field["options"] and len(field["labels"]) == len(field["options"])
    assert items["api_key"]["secret"] is True and "options" not in items["model"]
    assert items["extra_body"]["editor_mode"] is True and items["extra_body"]["default"] == ""
    defaults = {name: field["default"] for name, field in items.items()}
    assert defaults["base_url"] == comfyui.DEFAULT_BASE_URL and defaults["timeout"] == comfyui.DEFAULT_TIMEOUT
    assert {k: defaults[f"{k}_size"] for k in comfyui.DEFAULT_SIZES} == comfyui.DEFAULT_SIZES
    assert (defaults["steps"], defaults["cfg"], defaults["sampler_name"], defaults["scheduler"]) == \
        (comfyui.DEFAULT_STEPS, comfyui.DEFAULT_CFG, comfyui.DEFAULT_SAMPLER, comfyui.DEFAULT_SCHEDULER)
    assert defaults["prompt_mode"] == ComfyUIBackend.default_prompt_mode
    assert defaults["required_positive"] == comfyui.DEFAULT_REQUIRED_POSITIVE
    assert defaults["required_negative"] == comfyui.DEFAULT_REQUIRED_NEGATIVE

    # 用 schema 默认值生成的请求与空配置完全一致
    backend = ComfyUIBackend(defaults)
    assert backend.missing() == "" and backend.summary() == ComfyUIBackend({}).summary()
    graphs = []
    for section in (defaults, {}):
        for orientation in ("portrait", "landscape", "square"):
            fake = FakeComfy()
            run(section, fake, DrawRequest("1girl", "lowres", orientation, seed=11))
            graphs.append(graph_of(fake))
    assert graphs[:3] == graphs[3:]


def test_schema_edit_and_balance_fields_match_code():
    items = schema_items()
    keys = list(items)
    assert keys.index("edit_workflow_json") == keys.index("workflow_json") + 1
    field = items["edit_workflow_json"]
    assert field["type"] == "text" and field["default"] == "" and field["editor_mode"] is True
    assert field["editor_language"] == "json" and "secret" not in field
    assert "{{image}}" in field["hint"] and f"{{{{image{comfyui.MAX_EDIT_IMAGES}}}}}" in field["hint"]
    assert "{{denoise}}" in field["hint"] and "{{megapixels}}" in field["hint"]
    assert "修改图片" in items["model"]["hint"] and "余额" in items["api_key"]["hint"]
    assert "余额" in items["extra_body"]["hint"]
    # 没有新增只用于查询余额的密钥：Cloud 用 api_key，本地用 extra_body 中的 comfy.org Key
    assert [name for name, f in items.items() if f.get("secret")] == ["api_key"]

    defaults = {name: f["default"] for name, f in items.items()}
    backend = ComfyUIBackend(defaults)
    assert backend.edit_missing() == "" and backend.max_edit_images == 1 and backend.supports_strength is True
    assert not backend.custom_edit_workflow and "不计费" in backend.balance_missing()
    graphs, params = [], []
    for section in (defaults, {}):
        fake = FakeComfy()
        params.append(run(section, fake, edit_req(jpeg())).params)
        graphs.append(graph_of(fake))
    assert graphs[0] == graphs[1] and params[0]["size"] == params[1]["size"] == "1232x816"


def test_edit_not_blocked_by_broken_text_to_image_workflow():
    backend = ComfyUIBackend({"workflow_json": "{not json"})
    assert "JSON" in backend.missing()  # 文生图仍然报告工作流错误
    assert backend.edit_ready_missing() == "" and backend.edit_missing() == ""  # 修改图片不受影响


def test_plan_is_edit_aware():
    backend = ComfyUIBackend({})
    req = DrawRequest("x", orientation="landscape", images=[InputImage(b"\x89PNG", "image/png", 1600, 900)])
    plan = backend.plan(req)
    assert plan["mode"] == "edit" and plan["input_images"] == 1
    assert plan["size"] != backend.plan(DrawRequest("x", orientation="landscape"))["size"]
    unknown = backend.plan(DrawRequest("x", orientation="landscape", images=[InputImage(b"\x89PNG", "image/png")]))
    assert unknown["size"].endswith("MP")
