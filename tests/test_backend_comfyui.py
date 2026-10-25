"""ComfyUI 后端的单元测试（模拟 ComfyUI / Comfy Cloud，不访问网络）：python -m pytest tests"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import comfyui  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool  # noqa: E402
from backends.comfyui import ComfyUIBackend  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
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
                 cancel_api=True, view=PNG):
        self.pending = pending
        self.entry = OK_ENTRY if entry is None else entry
        self.checkpoints = checkpoints
        self.on_prompt = on_prompt
        self.in_queue = in_queue
        self.cancel_api = cancel_api
        self.view = view
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.accepted: list[str] = []
        self.polls = 0

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
        return httpx.Response(404, json={"error": "Not Found"})


def run(section, handler, req=None, *, action="generate"):
    async def go():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            backend = ComfyUIBackend(section)
            if action == "models":
                return await backend.list_models(Clients(http=pool))
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
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

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
        return httpx.Response(404, json={"code": "not_found", "message": "Not found"})


CLOUD = {"base_url": "https://cloud.comfy.org/api/", "api_key": "comfyui-key"}


def test_cloud_flow():
    fake = FakeCloud()
    result = run(CLOUD, fake, DrawRequest("1girl", "lowres", "square", seed=3))
    assert fake.paths() == ["/api/experiment/models/checkpoints", "/api/prompt", "/api/jobs/job-1",
                            "/api/jobs/job-1", "/api/jobs/job-1", "/api/view", "/out.png"]
    body = fake.bodies[0]
    assert set(body) == {"prompt"}  # Cloud 的请求体没有 prompt_id / client_id
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
