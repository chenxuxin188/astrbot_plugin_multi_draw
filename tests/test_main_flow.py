"""main.py / webapi.py 的端到端测试：用 astrbot_stub 模拟 AstrBot，所有 HTTP 请求走 MockTransport。"""

import asyncio
import base64
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PLUGIN_DIR.parent))

import astrbot_stub  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def make_zip(data=PNG):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image_0.png", data)
    return buf.getvalue()


def real_png(w=64, h=48):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (255, 0, 0, 128)).save(buf, "PNG")
    return buf.getvalue()


def defaults(schema):
    out = {}
    for name, field in schema.items():
        if field.get("type") == "object":
            out[name] = {k: v.get("default") for k, v in field["items"].items()}
        else:
            out[name] = field.get("default")
    return out


@pytest.fixture()
def env(tmp_path):
    astrbot_stub.install(tmp_path)
    for mod in list(sys.modules):
        if mod.startswith(PLUGIN_DIR.name):
            del sys.modules[mod]
    main = __import__(f"{PLUGIN_DIR.name}.main", fromlist=["*"])
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    config = astrbot_stub.AstrBotConfig(defaults(schema))
    config["admin_ids"] = ["10001"]
    config["novelai"]["token"] = "pst-secret-token-1234"
    config["novelai"]["track_anlas"] = False
    return main, config, tmp_path


def llm_responder(kwargs):
    system = kwargs.get("system_prompt") or ""
    if "Danbooru" in system or "JSON" in system:
        return json.dumps({"prompt": "1girl, cat ears", "negative": "", "orientation": "square"})
    if kwargs.get("image_urls"):
        return "看图评价：这只猫娘画得真可爱！"
    prompt = kwargs.get("prompt") or ""
    if "开始动笔" in prompt:
        return "收到，我开始画啦！"
    if "失败" in prompt:
        return "呜呜，失败了……"
    return "画好啦，满意吗？"


def make_plugin(main, config, handler, responder=llm_responder, providers=None):
    from importlib import import_module

    base = import_module(f"{PLUGIN_DIR.name}.backends.base")
    novelai = import_module(f"{PLUGIN_DIR.name}.backends.novelai")
    ctx = astrbot_stub.Context(responder, providers)
    plugin = main.MultiDrawPlugin(ctx, config)
    transport = httpx.MockTransport(handler)
    plugin.http = base.HttpPool(transport=transport)
    plugin.nai = novelai.NovelAIClient(transport=transport)
    plugin.clients = base.Clients(http=plugin.http, novelai=plugin.nai)
    return plugin, ctx


def texts(event):
    out = []
    for result in event.sent:
        for comp in result:
            out.append(getattr(comp, "text", None) or ("<image>" if hasattr(comp, "data") else repr(comp)))
    return out


def nai_ok(request):
    assert request.url.path == "/ai/generate-image"
    body = json.loads(request.content)
    assert "1girl, cat ears" in body["input"] and "very aesthetic" in body["input"]
    assert body["parameters"]["width"] == 1024  # LLM chose square
    return httpx.Response(200, content=make_zip())


def run(coro):
    return asyncio.run(coro)


def test_novelai_flow_text_comment(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)
    event = astrbot_stub.AstrMessageEvent()

    async def go():
        await plugin._draw_flow(event, "画一个猫娘", "", "llm")
        records, *_ = await plugin.audit.page(1, 10)
        await plugin.terminate()
        return records

    records = run(go())
    assert texts(event) == ["收到，我开始画啦！", "<image>", "画好啦，满意吗？"]
    rec = records[0]
    assert rec["status"] == "success" and rec["backend"] == "novelai" and rec["size"] == "1024x1024"
    assert rec["image"].count("/") == 1 and (tmp / "astrbot_plugin_multi_draw" / "images" / rec["image"]).is_file()
    assert not any(c.get("image_urls") for c in ctx.calls)


def test_comment_with_image(env):
    main, config, tmp = env
    config["persona"]["comment_with_image"] = True
    config["persona"]["vision_provider_id"] = "vision-llm"
    seen = {}

    def responder(kwargs):
        if kwargs.get("image_urls"):
            path = kwargs["image_urls"][0]
            seen["path"] = path
            seen["exists"] = os.path.isfile(path)
            seen["jpeg"] = Path(path).read_bytes()[:3] == b"\xff\xd8\xff"
            seen["pid"] = kwargs["chat_provider_id"]
            seen["prompt"] = kwargs["prompt"]
        return llm_responder(kwargs)

    def handler(request):
        return httpx.Response(200, content=make_zip(real_png()))

    plugin, ctx = make_plugin(main, config, handler, responder)
    event = astrbot_stub.AstrMessageEvent()
    run(plugin._draw_flow(event, "画一个猫娘", "", "llm"))
    assert texts(event)[-1] == "看图评价：这只猫娘画得真可爱！"
    assert seen["exists"] and seen["jpeg"] and seen["pid"] == "vision-llm"
    assert "附在这条消息里" in seen["prompt"]
    assert not os.path.exists(seen["path"])  # 临时文件已删除


def test_comment_with_image_falls_back_to_text(env):
    main, config, tmp = env
    config["persona"]["comment_with_image"] = True

    def responder(kwargs):
        if kwargs.get("image_urls"):
            raise RuntimeError("model does not support images")
        return llm_responder(kwargs)

    plugin, ctx = make_plugin(main, config, lambda r: httpx.Response(200, content=make_zip()), responder)
    event = astrbot_stub.AstrMessageEvent()
    run(plugin._draw_flow(event, "画一个猫娘", "", "llm"))
    assert texts(event)[-1] == "画好啦，满意吗？"
    assert any(c.get("image_urls") for c in ctx.calls)


def test_vision_skips_text_only_provider(env):
    main, config, tmp = env
    config["persona"]["comment_with_image"] = True
    config["persona"]["vision_provider_id"] = "text-only"

    class Prov:
        provider_config = {"modalities": ["text", "tool_use"]}

    plugin, ctx = make_plugin(main, config, lambda r: httpx.Response(200, content=make_zip()),
                              providers={"text-only": Prov()})
    event = astrbot_stub.AstrMessageEvent()
    run(plugin._draw_flow(event, "画一个猫娘", "", "llm"))
    image_calls = [c for c in ctx.calls if c.get("image_urls")]
    assert image_calls and all(c["chat_provider_id"] == "session-llm" for c in image_calls)


def test_generate_failure(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, lambda r: httpx.Response(401, json={"message": "Invalid token"}))
    event = astrbot_stub.AstrMessageEvent()

    async def go():
        await plugin._draw_flow(event, "画一个猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    out = texts(event)
    assert out[0] == "收到，我开始画啦！" and out[1].startswith("绘图失败：") and "Token" in out[1]
    assert out[2] == "呜呜，失败了……"
    assert rec["status"] == "failed" and rec["stage"] == "generate"


def test_backend_not_configured(env):
    main, config, tmp = env
    config["novelai"]["token"] = ""
    plugin, ctx = make_plugin(main, config, nai_ok)
    event = astrbot_stub.AstrMessageEvent()
    run(plugin._draw_flow(event, "画猫", "", "llm"))
    assert texts(event) == ["管理员还没有配置 NovelAI Token，暂时没法画画哦。"]


def test_queue_and_inflight(env):
    main, config, tmp = env
    gate = asyncio.Event()

    async def slow_handler(request):
        await gate.wait()
        return httpx.Response(200, content=make_zip())

    plugin, ctx = make_plugin(main, config, slow_handler)
    ev1, ev2, ev3 = (astrbot_stub.AstrMessageEvent(sender=s) for s in ("1", "2", "1"))

    async def go():
        t1 = asyncio.create_task(plugin._draw_flow(ev1, "画猫", "", "llm"))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(plugin._draw_flow(ev2, "画狗", "", "llm"))
        await asyncio.sleep(0.05)
        await plugin._draw_flow(ev3, "再画一张", "", "llm")  # 同一用户
        gate.set()
        await asyncio.gather(t1, t2)

    run(go())
    assert "（前面还有 1 张在排队）" in texts(ev2)[0]
    assert texts(ev3) == ["你的上一张画还没完成，请稍等一下～"]


def test_web_api(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, lambda r: httpx.Response(200, content=make_zip()))
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    assert set(routes) == {"state", "config", "test", "models", "balance", "audit", "usage", "images", "image"}
    req = astrbot_stub.request

    async def go():
        state = await routes["state"]()
        assert state["config"]["novelai"]["token"].endswith("1234")
        assert "pst-secret-token" not in json.dumps(state["config"])  # 完整的 Token 不能出现在页面数据中
        assert [b["key"] for b in state["backends"]][0] == "novelai"
        # 保存：掩码保持原值，修改并发数
        req.body = {"config": {"novelai": {"token": state["config"]["novelai"]["token"]},
                               "draw": {"max_concurrency": "3"}}}
        saved = await routes["config"]()
        assert saved["changed"] == ["draw.max_concurrency"] and config.saves == 1
        assert config["novelai"]["token"] == "pst-secret-token-1234"
        req.body = {"config": {"draw": {"backend": "nope"}}}
        bad = await routes["config"]()
        assert bad["status"] == "error" and "backend" in bad["message"]
        # 试画：页面上未保存的尺寸生效
        req.body = {"backend": "novelai", "section": {"square_size": "1024x1024"}, "prompt": "1girl",
                    "orientation": "square"}
        test = await routes["test"]()
        assert test["image"].startswith("data:image/png;base64,") and test["params"]["size"] == "1024x1024"
        req.body = {"backend": "novelai", "prompt": ""}
        assert (await routes["test"]())["status"] == "error"
        req.body = {"backend": "novelai"}
        assert "不支持" in (await routes["models"]())["message"]
        # 图库与记录
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画猫", "", "llm")
        req.query = {"limit": "10"}
        images = (await routes["images"]())["images"]
        req.query = {"name": images[0]["name"]}
        assert (await routes["image"]())["image"].startswith("data:image/png;base64,")
        req.query = {"name": "../../etc/passwd"}
        assert (await routes["image"]())["status"] == "error"
        req.query = {"page": "1"}
        assert (await routes["audit"]())["total"] == 1
        await plugin.terminate()
        assert ctx.registered_web_apis == []

    run(go())


def test_draw_use_command(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)

    async def collect(agen):
        return [r async for r in agen]

    admin = astrbot_stub.AstrMessageEvent(sender="10001")
    out = run(collect(plugin.draw_use(admin, "豆包")))
    assert config["draw"]["backend"] == "seedream" and config.saves == 1
    assert "已切换到" in out[0][0].text
    other = astrbot_stub.AstrMessageEvent(sender="999")
    out = run(collect(plugin.draw_use(other, "openai")))
    assert "权限不足" in out[0][0].text and config["draw"]["backend"] == "seedream"
    status = run(collect(plugin.draw_status(admin)))[0][0].text
    assert "当前绘图平台" in status and "seedream" in status


def test_openai_natural_flow(env):
    main, config, tmp = env
    config["draw"]["backend"] = "openai"
    config["openai"]["api_key"] = "sk-test-abcdefgh"
    seen = {}

    def responder(kwargs):
        system = kwargs.get("system_prompt") or ""
        if "image description" in system:
            seen["natural_system"] = True
            return json.dumps({"prompt": "A cat girl reading under cherry blossoms.", "negative": "text",
                               "orientation": "landscape"})
        return llm_responder(kwargs)

    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/images/generations"
        assert request.headers["authorization"] == "Bearer sk-test-abcdefgh"
        body = json.loads(request.content)
        seen["body"] = body
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    plugin, ctx = make_plugin(main, config, handler, responder)
    event = astrbot_stub.AstrMessageEvent()

    async def go():
        await plugin._draw_flow(event, "画一个在樱花树下看书的猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    assert seen.get("natural_system")
    body = seen["body"]
    assert body["model"] == config["openai"]["model"] and body["size"] == "1536x1024"
    assert "response_format" not in body and "A cat girl reading" in body["prompt"] and "Avoid: text" in body["prompt"]
    assert texts(event)[1] == "<image>" and rec["status"] == "success" and rec["backend"] == "openai"
    assert rec["size"] == "1536x1024"


def test_webchat_detaches_when_drawing_outlasts_tool_timeout(env):
    main, config, tmp = env
    gate = asyncio.Event()

    async def slow(request):
        await gate.wait()
        return httpx.Response(200, content=make_zip())

    plugin, ctx = make_plugin(main, config, slow)
    plugin._inline_wait_seconds = lambda event: 0.05
    main.DETACH_SETTLE_SECONDS = 0.05
    event = astrbot_stub.AstrMessageEvent(platform="webchat")

    async def go():
        assert await plugin.draw_image_tool(event, "画一个猫娘", "") is None
        assert event.get_extra(main.DETACHED_KEY)
        event.closed = True  # 本轮结束，WebChat 的回复通道关闭
        gate.set()
        await asyncio.gather(*list(plugin._tasks))

    run(go())
    assert texts(event) == ["收到，我开始画啦！"]
    late = [getattr(c, "text", None) or "<image>" for _, chain in ctx.proactive for c in chain]
    assert late == ["<image>", "画好啦，满意吗？"]
    assert all(session == event.unified_msg_origin for session, _ in ctx.proactive)


def test_webchat_waits_inline_when_fast(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)
    event = astrbot_stub.AstrMessageEvent(platform="webchat")
    run(plugin.draw_image_tool(event, "画一个猫娘", ""))
    assert texts(event) == ["收到，我开始画啦！", "<image>", "画好啦，满意吗？"] and not ctx.proactive


def test_inline_wait_reads_tool_call_timeout(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)
    event = astrbot_stub.AstrMessageEvent(platform="webchat")
    assert plugin._inline_wait_seconds(event) == 50  # 未配置时按旧版本默认 60 秒
    ctx.config = {"provider_settings": {"tool_call_timeout": 120}}
    assert plugin._inline_wait_seconds(event) == 110
    ctx.config = {"agent_runner": {"config": {"misc": {"tool_call_timeout": 300}}},
                  "provider_settings": {"tool_call_timeout": 60}}
    assert plugin._inline_wait_seconds(event) == 290
    ctx.config = {"provider_settings": {"tool_call_timeout": 3}}
    assert plugin._inline_wait_seconds(event) == 5


def test_failed_generation_records_planned_params(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, lambda r: httpx.Response(402, json={"message": "Not enough Anlas"}))

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    assert rec["status"] == "failed" and rec["stage"] == "generate"
    assert rec["model"] == "nai-diffusion-4-5-full" and rec["size"] == "1024x1024"
    assert rec["steps"] == 28 and rec["seed"] > 0 and rec["opus_free"] is True


def test_planned_seed_matches_sent_seed(env):
    main, config, tmp = env
    seen = {}

    def handler(request):
        seen["seed"] = json.loads(request.content)["parameters"]["seed"]
        return httpx.Response(200, content=make_zip())

    plugin, ctx = make_plugin(main, config, handler)

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    assert run(go())["seed"] == seen["seed"]


def test_persona_override_default_and_none(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)

    class PM:
        personas_v3 = [{"name": "catgirl", "prompt": "你是猫娘"}]

        def get_persona_v3_by_id(self, pid):
            return {"name": "default", "prompt": "内置默认人格"} if pid == "default" else None

    ctx.persona_manager = PM()
    event = astrbot_stub.AstrMessageEvent()
    config["persona"]["persona_id"] = "default"
    assert run(plugin._persona_prompt(event)) == "内置默认人格"
    config["persona"]["persona_id"] = "catgirl"
    assert run(plugin._persona_prompt(event)) == "你是猫娘"
    config["persona"]["persona_id"] = "[%None]"
    assert run(plugin._persona_prompt(event)) == ""


def test_account_masks_base_url_credentials(env):
    main, config, tmp = env
    config["novelai"]["base_url"] = "https://user:secretpw@relay.example"

    def handler(request):
        assert request.url.path == "/user/subscription"
        return httpx.Response(200, json={"tier": 3, "active": True})

    plugin, ctx = make_plugin(main, config, handler)

    async def collect():
        return [r async for r in plugin.draw_account(astrbot_stub.AstrMessageEvent(sender="10001"))]

    text = run(collect())[0][0].text
    assert "secretpw" not in text and "//***@relay.example" in text


def test_bad_backend_config_does_not_break_state(env):
    main, config, tmp = env
    config["seedream"]["base_url"] = "http://[::1"
    config["seedream"]["api_key"] = "ark-test-key"
    config["novelai"]["portrait_size"] = "9" * 400 + "x1"
    plugin, ctx = make_plugin(main, config, nai_ok)
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    state = run(routes["state"]())
    seedream = next(b for b in state["backends"] if b["key"] == "seedream")
    assert "地址" in seedream["missing"]
    assert next(b for b in state["backends"] if b["key"] == "novelai")["summary"].startswith("NovelAI")


def test_raising_concurrency_releases_queue(env):
    main, config, tmp = env
    config["draw"]["max_concurrency"] = 1
    plugin, ctx = make_plugin(main, config, nai_ok)
    backend = plugin.backend("openai")
    started = []

    async def job(i, gate):
        async with plugin.slot(backend):
            started.append(i)
            await gate.wait()

    async def go():
        gate = asyncio.Event()
        tasks = [asyncio.create_task(job(i, gate)) for i in range(3)]
        await asyncio.sleep(0.01)
        assert started == [0] and plugin._queued_ahead(backend) == 3
        config["draw"]["max_concurrency"] = 3
        plugin.wake_queues()
        await asyncio.sleep(0.01)
        assert started == [0, 1, 2]
        gate.set()
        await asyncio.gather(*tasks)

    run(go())


def test_prompt_stage_failure_records_model(env):
    main, config, tmp = env

    def responder(kwargs):
        if "Danbooru" in (kwargs.get("system_prompt") or ""):
            raise RuntimeError("LLM down")
        return llm_responder(kwargs)

    plugin, ctx = make_plugin(main, config, nai_ok, responder)

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    assert rec["stage"] == "prompt" and rec["model"] == "nai-diffusion-4-5-full"


def test_state_offers_builtin_default_persona(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok)

    class PM:
        personas_v3 = [{"name": "catgirl", "prompt": "你是猫娘"}]

    ctx.persona_manager = PM()
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    personas = run(routes["state"]())["personas"]
    assert personas[0]["name"] == "default" and [p["name"] for p in personas] == ["default", "catgirl"]


def test_test_endpoint_rejects_mask_with_appended_key(env):
    main, config, tmp = env
    config["openai"]["api_key"] = "sk-saved-key-12345678"
    plugin, ctx = make_plugin(main, config, nai_ok)
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    astrbot_stub.request.body = {"backend": "openai", "prompt": "a cat",
                                 "section": {"api_key": "••••5678sk-new"}}
    out = run(routes["test"]())
    assert out["status"] == "error" and "掩码" in out["message"]


# ---------------------------------------------------------------- 图片修改与余额


def edit_responder(kwargs):
    system = kwargs.get("system_prompt") or ""
    if "image-editing" in system:
        return json.dumps({"prompt": "Change the background to a beach. Keep the character unchanged.",
                           "negative": "", "orientation": ""})
    if "img2img" in system:
        return json.dumps({"prompt": "1girl, beach, smile", "negative": "", "orientation": "", "strength": 0.45})
    return llm_responder(kwargs)


def image_component(data: bytes):
    comp = astrbot_stub.Image("base64://" + base64.b64encode(data).decode())
    comp.data, comp.path, comp.url = data, "", ""
    return comp


def test_edit_image_from_message_with_separate_edit_backend(env):
    main, config, tmp = env
    config["draw"]["edit_backend"] = "openai"
    config["openai"]["api_key"] = "sk-test-abcdefgh"
    seen = {}

    def handler(request):
        assert request.url.path == "/v1/images/edits"
        seen["body"] = request.content
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(real_png()).decode()}]})

    plugin, ctx = make_plugin(main, config, handler, edit_responder)
    event = astrbot_stub.AstrMessageEvent()
    event.messages = [image_component(real_png(80, 40))]

    async def go():
        await plugin.edit_image_tool(event, "把背景换成海边")
        await asyncio.gather(*list(plugin._tasks))
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    assert texts(event)[1:] == ["<image>", "画好啦，满意吗？"]
    assert b'name="image"' in seen["body"] and b"Change the background to a beach" in seen["body"]
    assert rec["status"] == "success" and rec["mode"] == "edit" and rec["backend"] == "openai"
    assert rec["input_images"] == 1 and rec["orientation"] == "landscape"  # 保持原图方向
    prompt_calls = [c for c in ctx.calls if "image-editing" in (c.get("system_prompt") or "")]
    assert prompt_calls and prompt_calls[0].get("image_urls")  # 原图一起发给了提示词 LLM
    start = [c for c in ctx.calls if "开始动笔" in (c.get("prompt") or "")]
    assert start and "修改对方发来的图片" in start[0]["prompt"]


def test_edit_last_drawn_image_with_novelai_img2img(env):
    main, config, tmp = env
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, content=make_zip(real_png()))

    plugin, ctx = make_plugin(main, config, handler, edit_responder)

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        event = astrbot_stub.AstrMessageEvent()
        await plugin.edit_image_tool(event, "把背景换成海边")  # 没有发图：修改刚才画的那张
        await asyncio.gather(*list(plugin._tasks))
        return event, (await plugin.audit.page(1, 10))[0][0]

    event, rec = run(go())
    assert bodies[0]["action"] == "generate" and bodies[1]["action"] == "img2img"
    assert bodies[1]["parameters"]["strength"] == 0.45 and bodies[1]["parameters"]["image"]
    assert rec["mode"] == "edit" and rec["status"] == "success"


def test_edit_reports_unreadable_or_missing_images(env):
    main, config, tmp = env
    plugin, ctx = make_plugin(main, config, nai_ok, edit_responder)
    broken = astrbot_stub.Image("https://multimedia.nt.qq.com.cn/download?rkey=x")
    broken.url, broken.path = broken.file, ""
    event = astrbot_stub.AstrMessageEvent()
    event.messages = [broken]
    plugin.http = type(plugin.http)(transport=httpx.MockTransport(lambda r: httpx.Response(400)))

    async def go():
        await plugin.edit_image_tool(event, "改成横图")
        await asyncio.gather(*list(plugin._tasks))
        empty = astrbot_stub.AstrMessageEvent(sender="2")
        await plugin.edit_image_tool(empty, "改成横图")
        await asyncio.gather(*list(plugin._tasks))
        return empty

    empty = run(go())
    assert "读取失败" in texts(event)[0]
    assert "没有找到要修改的图片" in texts(empty)[0]


def test_balance_command_and_web(env):
    main, config, tmp = env

    def handler(request):
        assert request.url.path == "/user/subscription"
        return httpx.Response(200, json={"tier": 3, "active": True, "trainingStepsLeft": {
            "fixedTrainingStepsLeft": 10000, "purchasedTrainingSteps": 500}})

    plugin, ctx = make_plugin(main, config, handler)

    async def collect(agen):
        return [r async for r in agen]

    admin = astrbot_stub.AstrMessageEvent(sender="10001")
    text = run(collect(plugin.draw_balance(admin, "")))[0][0].text
    assert "【NovelAI】" in text and "10500" in text
    text = run(collect(plugin.draw_balance(admin, "gemini")))[0][0].text
    assert "【Gemini" in text and "aistudio.google.com" in text
    assert "权限不足" in run(collect(plugin.draw_balance(astrbot_stub.AstrMessageEvent(sender="9"), "")))[0][0].text

    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    astrbot_stub.request.body = {"backend": "novelai"}
    out = run(routes["balance"]())
    assert any("10500" in v for _, v in out["items"])
    astrbot_stub.request.body = {"backend": "gemini"}
    assert run(routes["balance"]())["status"] == "error"


def test_web_test_endpoint_edit(env):
    main, config, tmp = env
    config["openai"]["api_key"] = "sk-test-abcdefgh"

    def handler(request):
        assert request.url.path == "/v1/images/edits"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(real_png()).decode()}]})

    plugin, ctx = make_plugin(main, config, handler)
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    data_url = "data:image/png;base64," + base64.b64encode(real_png(40, 80)).decode()
    astrbot_stub.request.body = {"backend": "openai", "prompt": "make it night", "images": [data_url],
                                 "orientation": ""}
    out = run(routes["test"]())
    assert out["image"].startswith("data:image/") and out["params"]["mode"] == "edit"
    astrbot_stub.request.body = {"backend": "openai", "prompt": "x", "images": [data_url] * 17}
    assert run(routes["test"]())["status"] == "error"


# ---------------------------------------------------------------- v1.1 审查后补充的回归测试


def test_image_read_timeout_never_edits_last_drawing(env):
    main, config, tmp = env
    actions = []

    def handler(request):
        actions.append(json.loads(request.content)["action"])
        return httpx.Response(200, content=make_zip(real_png()))

    plugin, ctx = make_plugin(main, config, handler, edit_responder)

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=real_png())

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        main.IMAGE_FETCH_TIMEOUT = 0.2
        plugin.http = type(plugin.http)(transport=httpx.MockTransport(slow))
        stalled = astrbot_stub.Image("https://multimedia.nt.qq.com.cn/download?rkey=x")
        stalled.url, stalled.path = stalled.file, ""
        event = astrbot_stub.AstrMessageEvent()
        event.messages = [stalled]
        await plugin.edit_image_tool(event, "把背景换成海边")
        await asyncio.gather(*list(plugin._tasks))
        return event

    event = run(go())
    assert actions == ["generate"]  # 没有去修改上一张画
    assert "重新发送" in texts(event)[0]


def test_edit_uses_image_from_users_previous_message(env):
    main, config, tmp = env
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=make_zip(real_png()))

    plugin, ctx = make_plugin(main, config, handler, edit_responder)
    photo = real_png(120, 60)

    async def go():
        # 用户先单独发了一张图（没有唤醒 Bot，只被监听器记下），再发文字要求修改
        sent = astrbot_stub.AstrMessageEvent(sender="42")
        sent.messages = [image_component(photo)]
        await plugin.remember_user_images(sent)
        ask = astrbot_stub.AstrMessageEvent(sender="42")
        await plugin.edit_image_tool(ask, "把它改成夏天")
        await asyncio.gather(*list(plugin._tasks))

    run(go())
    assert bodies and bodies[-1]["action"] == "img2img"
    assert bodies[-1]["parameters"]["width"] > bodies[-1]["parameters"]["height"]  # 用的是用户发的横图


def test_unresolved_quote_is_not_replaced_by_last_drawing(env):
    main, config, tmp = env
    actions = []

    def handler(request):
        actions.append(json.loads(request.content)["action"])
        return httpx.Response(200, content=make_zip(real_png()))

    plugin, ctx = make_plugin(main, config, handler, edit_responder)

    async def go():
        await plugin._draw_flow(astrbot_stub.AstrMessageEvent(), "画一个猫娘", "", "llm")
        reply = astrbot_stub.Reply([])  # 平台没能取到被引用的消息
        reply.id = "999"
        event = astrbot_stub.AstrMessageEvent(platform="discord")
        event.messages = [reply]
        event.get_self_id = lambda: "bot"
        await plugin.edit_image_tool(event, "改成夏天")
        await asyncio.gather(*list(plugin._tasks))
        return event

    event = run(go())
    assert actions == ["generate"] and "重新发送" in texts(event)[0]


def test_tag_edit_keeps_source_orientation(env):
    main, config, tmp = env
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=make_zip(real_png()))

    def responder(kwargs):
        if "img2img" in (kwargs.get("system_prompt") or ""):
            # LLM 不听话地给了一个方向
            return json.dumps({"prompt": "1girl, beach", "negative": "", "orientation": "portrait", "strength": 0.5})
        return llm_responder(kwargs)

    plugin, ctx = make_plugin(main, config, handler, responder)
    event = astrbot_stub.AstrMessageEvent()
    event.messages = [image_component(real_png(160, 90))]

    async def go():
        await plugin.edit_image_tool(event, "把背景换成海边")
        await asyncio.gather(*list(plugin._tasks))

    run(go())
    params = bodies[-1]["parameters"]
    assert params["width"] > params["height"]  # 保持原图的横向比例，没有被裁成竖图
    tag_prompt = [c["system_prompt"] for c in ctx.calls if "img2img" in (c.get("system_prompt") or "")][0]
    assert 'Rules for "orientation"' not in tag_prompt


def test_state_reports_edit_prompt_mode(env):
    main, config, tmp = env
    config["comfyui"]["edit_prompt_mode"] = "natural"
    plugin, ctx = make_plugin(main, config, nai_ok)
    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    comfy = next(b for b in run(routes["state"]())["backends"] if b["key"] == "comfyui")
    assert comfy["prompt_mode"] == "tags" and comfy["edit_prompt_mode"] == "natural"


def openai_usage_handler(usage):
    def handler(request):
        assert request.url.path == "/v1/images/generations"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}], "usage": usage})

    return handler


async def collect_results(agen):
    return [r async for r in agen]


def test_usage_recorded_with_token_price_and_summarized(env):
    main, config, tmp = env
    config["draw"]["backend"] = "openai"
    config["openai"].update(api_key="sk-test-abcdefgh", price_input_per_million=5.0, price_output_per_million=40.0)
    usage = {"input_tokens": 100, "output_tokens": 4160, "total_tokens": 4260}
    plugin, ctx = make_plugin(main, config, openai_usage_handler(usage))
    alice = astrbot_stub.AstrMessageEvent(sender="10001")
    bob = astrbot_stub.AstrMessageEvent(sender="20002")

    async def go():
        await plugin._draw_flow(alice, "画一只猫", "", "llm")
        await plugin._draw_flow(bob, "画一只狗", "", "llm")
        return (await plugin.audit.page(1, 10))[0]

    records = run(go())
    rec = records[0]
    assert rec["usage"] == {"images": 1, "input_tokens": 100, "output_tokens": 4160, "total_tokens": 4260}
    assert rec["cost"] == 0.16690 and rec["currency"] == "USD" and rec["cost_estimated"] is True
    assert "4,260 tokens" in main.format_record(rec) and "约 0.1669 USD" in main.format_record(rec)

    admin_text = run(collect_results(plugin.draw_usage(alice, "7")))[0][0].text
    assert "最近 7 天" in admin_text and "共 2 次绘图" in admin_text and "成功 2 张" in admin_text
    assert "8,520 tokens" in admin_text and "约 0.3338 USD" in admin_text
    own = run(collect_results(plugin.draw_usage(bob, "all", "openai")))[0][0].text
    assert own.startswith("你的OpenAI") and "全部记录" in own and "共 1 次绘图" in own
    bad = run(collect_results(plugin.draw_usage(bob, "whatever")))[0][0].text
    assert "无法识别" in bad

    # LLM 工具：管理员看到所有人，非管理员只看到自己的，且没有余额
    text = run(plugin.draw_usage_tool(alice, days=1))
    assert "今天" in text and "共 2 次绘图" in text
    text = run(plugin.draw_usage_tool(bob, days=1, platform="gpt"))
    assert "共 1 次绘图" in text and "只有管理员" in text
    assert "没有名为" in run(plugin.draw_usage_tool(alice, days=1, platform="midjourney"))

    routes = {r[0].rsplit("/", 1)[-1]: r[1] for r in ctx.registered_web_apis}
    astrbot_stub.request.query = {"days": "0"}
    out = run(routes["usage"]())
    assert out["count"] == 2 and out["label"] == "全部记录" and out["since"] == ""
    row = out["backends"][0]
    assert row["backend"] == "openai" and row["images"] == 2 and row["costs"] == {"USD": 0.3338}
    assert row["estimated"] == ["USD"] and out["total"]["total_tokens"] == 8520
    astrbot_stub.request.query = {"backend": "nope"}
    assert run(routes["usage"]())["status"] == "error"
    astrbot_stub.request.query = {}
    # 试画也显示用量和按（未保存的）设置估算的费用
    astrbot_stub.request.body = {"backend": "openai", "prompt": "a cat",
                                 "section": {"price_input_per_million": 10, "price_output_per_million": 50}}
    out = run(routes["test"]())
    assert out["params"]["usage"]["total_tokens"] == 4260 and out["cost"] == "约 0.209 USD"
    run(plugin.terminate())


def test_usage_off_and_per_image_price(env):
    main, config, tmp = env
    config["draw"]["backend"] = "openai"
    config["openai"].update(api_key="sk-test-abcdefgh", track_usage=False, price_per_image=0.2)
    plugin, ctx = make_plugin(main, config, openai_usage_handler({"total_tokens": 900}))
    event = astrbot_stub.AstrMessageEvent()

    async def go():
        await plugin._draw_flow(event, "画一只猫", "", "llm")
        config["openai"]["track_usage"] = True
        config["openai"]["price_currency"] = "元"
        await plugin._draw_flow(event, "画一只狗", "", "llm")
        return (await plugin.audit.page(1, 10))[0]

    newest, oldest = run(go())
    assert "usage" not in oldest and "cost" not in oldest
    # 只有总 token 数、没有输入 / 输出 token 时按张估算
    assert newest["usage"] == {"images": 1, "total_tokens": 900}
    assert newest["cost"] == 0.2 and newest["currency"] == "元" and newest["cost_estimated"] is True
    text = run(plugin.usage_text(0))
    assert "成功 2 张" in text and "约 0.2 元" in text


def test_anlas_tracking_setting_in_novelai_section(env, monkeypatch):
    main, config, tmp = env
    monkeypatch.setattr(main, "ANLAS_SETTLE_SECONDS", 0)
    config["novelai"]["track_anlas"] = True
    balance = {"anlas": 1000}

    def handler(request):
        if request.url.path == "/user/subscription":
            return httpx.Response(200, json={"tier": 3, "active": True, "trainingStepsLeft": {
                "fixedTrainingStepsLeft": balance["anlas"], "purchasedTrainingSteps": 0}})
        balance["anlas"] -= 17
        return nai_ok(request)

    plugin, ctx = make_plugin(main, config, handler)
    event = astrbot_stub.AstrMessageEvent()

    async def go():
        await plugin._draw_flow(event, "画一个猫娘", "", "llm")
        return (await plugin.audit.page(1, 10))[0][0]

    rec = run(go())
    assert rec["anlas_before"] == 1000 and rec["anlas_after"] == 983 and rec["anlas_cost"] == 17
    assert "usage" not in rec
    text = run(plugin.usage_text(1))
    assert "【NovelAI】成功 1 张；消耗 17 Anlas" in text

    run(plugin.terminate())

