"""NovelAI 后端（图生图修改图片、余额查询）的单元测试（不联网）：python -m pytest tests/test_backend_novelai.py"""

import asyncio
import base64
import io
import json
import math
import random
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import novelai as nai  # noqa: E402
from backends.base import BalanceInfo, Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.novelai import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_SIZES,
    EDIT_DEFAULT_STRENGTH,
    NAI_MAX_PIXELS,
    OPUS_FREE_MAX_PIXELS,
    NovelAIBackend,
    NovelAIClient,
    NovelAIError,
    edit_size,
    prepare_init_image,
)

Image = pytest.importorskip("PIL.Image")

TOKEN = "pst-secret-token-abcdef1234"
OUT_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
GEN_URL = "https://image.novelai.net/ai/generate-image"
SUB_URL = "https://image.novelai.net/user/subscription"


def make_zip(data: bytes = OUT_PNG) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image_0.png", data)
    return buf.getvalue()


def make_image(width: int, height: int, *, fmt: str = "PNG", mode: str = "RGB",
               color=(200, 30, 30), transparent_corner: bool = False) -> InputImage:
    im = Image.new(mode, (width, height), color)
    if transparent_corner:  # 左上角 1/4 区域完全透明
        im.paste((0, 0, 0, 0), (0, 0, width // 4, height // 4))
    buf = io.BytesIO()
    im.save(buf, fmt)
    return InputImage(buf.getvalue(), f"image/{fmt.lower()}", width, height)


def decode_sent_image(body: dict):
    raw = body["parameters"]["image"]
    assert isinstance(raw, str) and not raw.startswith("data:")
    data = base64.b64decode(raw, validate=True)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(data))


class Recorder:
    """记录请求并按顺序返回预设响应（响应可以是 httpx.Response、异常或函数）。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        resp = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            return resp(request)
        return resp

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def run(backend: NovelAIBackend, handler, req: DrawRequest | None = None, method: str = "generate"):
    async def main():
        transport = httpx.MockTransport(handler)
        pool = HttpPool(transport=transport)
        client = NovelAIClient(transport=transport)
        try:
            clients = Clients(http=pool, novelai=client)
            if method == "generate":
                return await backend.generate(clients, req or DrawRequest(prompt="1girl"))
            return await backend.balance(clients)
        finally:
            await client.close()
            await pool.close()

    return asyncio.run(main())


@pytest.fixture
def sleeps(monkeypatch):
    """跳过重试等待，并记录等待时间。"""
    delays = []

    async def fake_sleep(delay, *args, **kwargs):
        delays.append(delay)

    monkeypatch.setattr(nai.asyncio, "sleep", fake_sleep)
    return delays


def backend(**section) -> NovelAIBackend:
    return NovelAIBackend({"token": TOKEN, **section})


def edit_req(*images: InputImage, **kw) -> DrawRequest:
    kw.setdefault("orientation", "portrait")
    return DrawRequest(prompt="1girl, red dress", negative="lowres", images=list(images), **kw)


# ---------------------------------------------------------------- 能力与配置


def test_class_attributes_and_edit_missing(monkeypatch):
    b = backend()
    assert (b.max_edit_images, b.supports_strength, b.supports_balance) == (1, True, True)
    assert b.billing_url == "https://novelai.net/image"
    assert b.edit_missing() == "" and b.balance_missing() == ""
    for model in ("nai-diffusion-3", "nai-diffusion-4-5-curated", "nai-diffusion-5-full"):
        assert backend(model=model).edit_missing() == ""
    reason = backend(model="nai-diffusion-4-5-full-inpainting").edit_missing()
    assert "nai-diffusion-4-5-full-inpainting" in reason and "局部重绘" in reason
    # 没有 Pillow 时不能调整原图尺寸
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert "Pillow" in backend().edit_missing()


def test_balance_missing_without_token():
    reason = NovelAIBackend({}).balance_missing()
    assert "NovelAI Token" in reason
    with pytest.raises(DrawError) as info:
        run(NovelAIBackend({}), Recorder(httpx.Response(200, json={})), method="balance")
    assert "NovelAI Token" in str(info.value)


def test_edit_noise_setting():
    assert backend().edit_noise == 0.0
    assert backend(edit_noise=0.35).edit_noise == 0.35
    assert backend(edit_noise="5").edit_noise == 0.99
    assert backend(edit_noise=-1).edit_noise == 0.0
    assert backend(edit_noise="bad").edit_noise == 0.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.55, 0.55), (0, EDIT_DEFAULT_STRENGTH), (-1, EDIT_DEFAULT_STRENGTH), (float("nan"), EDIT_DEFAULT_STRENGTH),
     (1.5, 0.99), (0.001, 0.01), (0.123456, 0.12), ("0.4", 0.4)],
)
def test_clamp_strength(value, expected):
    assert nai.clamp_strength(value) == expected


# ---------------------------------------------------------------- 尺寸


def test_edit_size_examples():
    assert edit_size(1000, 1500, 832 * 1216) == (832, 1216)
    assert edit_size(1024, 1024, 1024 * 1024) == (1024, 1024)
    assert edit_size(1920, 1080, 1216 * 832) == (1280, 704)
    # 极端宽高比限制在 1:4 ~ 4:1，且不超过像素预算
    assert edit_size(10, 5000, OPUS_FREE_MAX_PIXELS) == (512, 2048)
    w, h = edit_size(5000, 10, OPUS_FREE_MAX_PIXELS)
    assert w / h == 4 and w * h <= OPUS_FREE_MAX_PIXELS
    assert edit_size(64, 64, 1) == (64, 64)
    # 预算不会超过官方上限
    w, h = edit_size(1000, 1000, 10 ** 9)
    assert w * h <= NAI_MAX_PIXELS


def test_edit_size_properties():
    rng = random.Random(42)
    for _ in range(3000):
        src_w, src_h = rng.randint(1, 4096), rng.randint(1, 4096)
        budget = rng.choice([OPUS_FREE_MAX_PIXELS, NAI_MAX_PIXELS, 832 * 1216, rng.randint(4096, 4 * 10 ** 6)])
        w, h = edit_size(src_w, src_h, budget)
        assert w % 64 == 0 and h % 64 == 0 and w >= 64 and h >= 64
        assert w * h <= min(max(budget, 4096), NAI_MAX_PIXELS)
        assert max(w / h, h / w) <= (4.2 if min(w, h) >= 512 else 4 * (1 + 64 / min(w, h)))
        # 宽高比在 1:4 ~ 4:1 内时，与原图相差不大
        ratio = src_w / src_h
        if 0.25 <= ratio <= 4 and budget >= 512 * 512:
            assert abs(math.log((w / h) / ratio)) < 0.1


def _best_edit_size(src_w: int, src_h: int, budget: int) -> tuple[int, int]:
    """穷举所有 64 倍数的尺寸，按 edit_size 的规则选最优（宽高比在容差内时面积最大）。"""
    budget = max(4096, min(budget, NAI_MAX_PIXELS))
    target = math.log(min(max(src_w / src_h, 1 / nai.EDIT_MAX_ASPECT), nai.EDIT_MAX_ASPECT))
    best = None
    for w in range(64, budget // 64 + 1, 64):
        for h in range(64, budget // w + 1, 64):
            key = (max(0.0, abs(math.log(w / h) - target) - nai.ASPECT_TOLERANCE), -w * h)
            if best is None or key < best[0]:
                best = (key, (w, h))
    return best[1]


@pytest.mark.parametrize(("src_w", "src_h", "budget"), [
    (334, 741, NAI_MAX_PIXELS), (1412, 3962, NAI_MAX_PIXELS), (984, 2480, NAI_MAX_PIXELS),
    (1000, 1500, 832 * 1216), (1920, 1080, 1216 * 832), (3000, 700, OPUS_FREE_MAX_PIXELS),
])
def test_edit_size_is_optimal_for_large_budgets(src_w, src_h, budget):
    # 高度超过约 2100 时 3% 的容差内有两个 64 的台阶，只试 h0 / h0+64 会选到偏小的尺寸
    w, h = edit_size(src_w, src_h, budget)
    best = _best_edit_size(src_w, src_h, budget)
    assert w * h == best[0] * best[1], ((w, h), best)


def test_edit_target_size_rules():
    portrait = make_image(1000, 1500)
    # 方向与原图一致：保持原图比例，像素数不超过配置尺寸
    assert backend().edit_target_size(edit_req(portrait), portrait) == (832, 1216)
    # 用户要求改成横图：使用横图配置尺寸（原图居中裁剪）
    assert backend().edit_target_size(edit_req(portrait, orientation="landscape"), portrait) == (1216, 832)
    # 不知道原图尺寸：使用配置尺寸
    unknown = InputImage(portrait.data, "image/png")
    assert backend().edit_target_size(edit_req(unknown), unknown) == (832, 1216)
    # 免费参数保护：即使配置了大尺寸也不超过 1024×1024
    w, h = backend(portrait_size="1536x2048").edit_target_size(edit_req(portrait), portrait)
    assert w * h <= OPUS_FREE_MAX_PIXELS
    # 关闭保护：可以使用更大的尺寸，但不超过官方上限 3145728
    w, h = backend(opus_free_guard=False, portrait_size="1536x2048").edit_target_size(edit_req(portrait), portrait)
    assert OPUS_FREE_MAX_PIXELS < w * h <= NAI_MAX_PIXELS and abs(w / h - 1000 / 1500) < 0.03
    w, h = backend(opus_free_guard=False, landscape_size="4096x4096").edit_target_size(
        edit_req(portrait, orientation="landscape"), portrait)
    assert w * h <= NAI_MAX_PIXELS and w % 64 == 0 and h % 64 == 0


# ---------------------------------------------------------------- 底图处理


def test_prepare_init_image_flattens_alpha_for_non_v5():
    src = make_image(1000, 1500, mode="RGBA", color=(10, 20, 30, 255), transparent_corner=True)
    out = Image.open(io.BytesIO(prepare_init_image(src.data, 832, 1216, keep_alpha=False)))
    assert out.format == "PNG" and out.size == (832, 1216) and out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((400, 600)) == (10, 20, 30)


def test_prepare_init_image_keeps_alpha_for_v5():
    src = make_image(1000, 1500, mode="RGBA", color=(10, 20, 30, 255), transparent_corner=True)
    out = Image.open(io.BytesIO(prepare_init_image(src.data, 832, 1216, keep_alpha=True)))
    assert out.size == (832, 1216) and out.mode == "RGBA" and out.getpixel((0, 0))[3] == 0


def test_prepare_init_image_center_crops_different_aspect():
    # 左右两半颜色不同的横图裁成方图：保留中间，两边被裁掉
    im = Image.new("RGB", (2000, 1000), (255, 0, 0))
    im.paste((0, 0, 255), (1000, 0, 2000, 1000))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    out = Image.open(io.BytesIO(prepare_init_image(buf.getvalue(), 512, 512, keep_alpha=False)))
    assert out.size == (512, 512) and out.mode == "RGB"
    left, right = out.getpixel((10, 256)), out.getpixel((500, 256))
    assert left[0] > 200 and left[2] < 60 and right[2] > 200 and right[0] < 60


def test_prepare_init_image_errors(monkeypatch):
    with pytest.raises(NovelAIError) as info:
        prepare_init_image(b"not an image", 64, 64, keep_alpha=False)
    assert "无法读取要修改的图片" in str(info.value)
    data = make_image(64, 64).data
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(NovelAIError) as info:
        prepare_init_image(data, 64, 64, keep_alpha=False)
    assert "Pillow" in str(info.value)


# ---------------------------------------------------------------- 修改图片请求


def test_edit_request_shape():
    rec = Recorder(httpx.Response(201, content=make_zip()))
    src = make_image(1000, 1500, mode="RGBA", transparent_corner=True)
    result = run(backend(), rec, edit_req(src, strength=0.55, seed=12345))

    assert len(rec.requests) == 1
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == GEN_URL
    assert request.headers["content-type"] == "application/json"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body = rec.body()
    assert set(body) == {"input", "model", "action", "parameters"}
    assert body["action"] == "img2img" and body["model"] == "nai-diffusion-4-5-full"
    assert body["input"] == "1girl, red dress"
    p = body["parameters"]
    assert (p["width"], p["height"]) == (832, 1216)
    assert p["strength"] == 0.55 and p["noise"] == 0.0
    assert p["seed"] == 12345 and p["extra_noise_seed"] == 12344
    assert p["color_correct"] is False and p["sm"] is False and p["sm_dyn"] is False
    assert p["n_samples"] == 1 and p["steps"] == 28
    assert "img2img" not in p and "mask" not in p  # 嵌套的 img2img 只用于局部重绘
    assert p["v4_prompt"]["caption"]["base_caption"] == "1girl, red dress"
    assert p["v4_negative_prompt"]["caption"]["base_caption"] == "lowres"
    sent = decode_sent_image(body)
    assert sent.size == (832, 1216) and sent.mode == "RGB"  # V4.5 不支持透明图，铺白底
    assert sent.getpixel((0, 0)) == (255, 255, 255)

    assert result.image == OUT_PNG and result.mime == "image/png"
    assert result.params == {
        "model": "nai-diffusion-4-5-full", "size": "832x1216", "steps": 28, "scale": 5.0,
        "sampler": "k_euler_ancestral", "seed": 12345, "opus_free": False,
        "mode": "edit", "input_images": 1, "strength": 0.55, "noise": 0.0,
    }
    assert TOKEN not in json.dumps(result.params)


def test_edit_v5_keeps_transparency_and_landscape_jpeg():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    src = make_image(800, 1200, mode="RGBA", transparent_corner=True)
    run(backend(model="nai-diffusion-5-full", edit_noise=0.2), rec, edit_req(src, strength=0.4))
    p = rec.body()["parameters"]
    assert rec.body()["model"] == "nai-diffusion-5-full" and p["noise"] == 0.2
    sent = decode_sent_image(rec.body())
    assert sent.mode == "RGBA" and sent.getpixel((0, 0))[3] == 0

    rec = Recorder(httpx.Response(200, content=make_zip()))
    jpeg = make_image(1920, 1080, fmt="JPEG")
    result = run(backend(), rec, edit_req(jpeg, orientation="landscape"))
    p = rec.body()["parameters"]
    assert (p["width"], p["height"]) == (1280, 704) and result.params["size"] == "1280x704"
    assert decode_sent_image(rec.body()).size == (1280, 704)
    # 没有给出重绘幅度时使用官方默认值
    assert p["strength"] == EDIT_DEFAULT_STRENGTH and result.params["strength"] == EDIT_DEFAULT_STRENGTH


def test_edit_orientation_change_and_unknown_size():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    src = make_image(800, 1200)
    result = run(backend(), rec, edit_req(src, orientation="landscape"))
    assert result.params["size"] == "1216x832"
    assert decode_sent_image(rec.body()).size == (1216, 832)

    rec = Recorder(httpx.Response(200, content=make_zip()))
    unknown = InputImage(make_image(600, 600).data, "image/png")
    result = run(backend(), rec, edit_req(unknown, orientation="square"))
    assert result.params["size"] == "1024x1024"
    assert decode_sent_image(rec.body()).size == (1024, 1024)


def test_edit_uses_only_first_image():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    first = make_image(640, 640, color=(255, 0, 0))
    second = make_image(640, 640, color=(0, 0, 255))
    result = run(backend(), rec, edit_req(first, second, orientation="square"))
    assert len(rec.requests) == 1
    assert decode_sent_image(rec.body()).getpixel((10, 10))[0] > 200
    assert result.params["input_images"] == 1 and result.params["ignored_images"] == 1


def test_edit_guard_off_allows_more_pixels_and_steps():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    b = backend(opus_free_guard=False, portrait_size="1536x2048", steps=40)
    result = run(b, rec, edit_req(make_image(1000, 1500)))
    p = rec.body()["parameters"]
    assert p["steps"] == 40 and OPUS_FREE_MAX_PIXELS < p["width"] * p["height"] <= NAI_MAX_PIXELS
    assert p["width"] % 64 == 0 and p["height"] % 64 == 0
    assert result.params["opus_free"] is False


def test_plan_matches_generate_without_image_processing():
    b = backend()
    req = edit_req(InputImage(b"not decoded during plan", "image/png", 1000, 1500), strength=0.3, seed=99)
    plan = b.plan(req)
    assert plan["mode"] == "edit" and plan["size"] == "832x1216" and plan["strength"] == 0.3
    assert plan["seed"] == 99 and plan["opus_free"] is False and plan["input_images"] == 1


def test_text_to_image_payload_unchanged():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    result = run(backend(), rec, DrawRequest(prompt="1girl", orientation="square", seed=7, strength=0.5))
    body = rec.body()
    assert body["action"] == "generate"
    for key in ("image", "strength", "noise", "extra_noise_seed", "color_correct"):
        assert key not in body["parameters"]
    assert "mode" not in result.params and result.params["opus_free"] is True
    assert result.params["size"] == "1024x1024"


# ---------------------------------------------------------------- 错误


def test_edit_refused_for_inpainting_model_without_request():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    with pytest.raises(DrawError) as info:
        run(backend(model="nai-diffusion-3-inpainting"), rec, edit_req(make_image(64, 64)))
    assert "局部重绘" in str(info.value) and not rec.requests


def test_edit_bad_image_not_sent():
    rec = Recorder(httpx.Response(200, content=make_zip()))
    with pytest.raises(NovelAIError) as info:
        run(backend(), rec, edit_req(InputImage(b"garbage", "image/png", 100, 100)))
    assert "无法读取要修改的图片" in str(info.value) and not rec.requests


def test_edit_402_explains_anlas_and_is_not_retried(sleeps):
    rec = Recorder(httpx.Response(402, json={"statusCode": 402, "message": "Not enough training steps"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(make_image(64, 64)))
    text = str(info.value)
    assert info.value.status == 402 and "Anlas 不足" in text and "Not enough Anlas" in text
    assert "修改图片" in text and "training steps" not in text
    assert len(rec.requests) == 1 and not sleeps and TOKEN not in text


@pytest.mark.parametrize("status", [500, 504, 524])
def test_edit_server_errors_not_retried(status, sleeps):
    rec = Recorder(httpx.Response(status, json={"statusCode": status, "message": "boom"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(make_image(64, 64)))
    assert info.value.status == status and not info.value.retryable
    assert len(rec.requests) == 1 and not sleeps


def test_edit_timeout_not_retried(sleeps):
    rec = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(make_image(64, 64)))
    assert "超时" in str(info.value) and len(rec.requests) == 1 and not sleeps


def test_edit_concurrent_lock_retried(sleeps):
    rec = Recorder(httpx.Response(429, json={"message": "Concurrent generation is locked"}),
                   httpx.Response(200, content=make_zip()))
    result = run(backend(), rec, edit_req(make_image(64, 64)))
    assert result.params["mode"] == "edit" and len(rec.requests) == 2 and sleeps == [5]
    # 重试时发送的是同一个请求（同一张底图和种子）
    assert rec.body(0) == rec.body(1)


def test_edit_413_and_400_messages():
    rec = Recorder(httpx.Response(413, text="<html><title>413 Request Entity Too Large</title></html>"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(make_image(64, 64)))
    assert "请求内容过大" in str(info.value) and "<" not in str(info.value)
    rec = Recorder(httpx.Response(400, json={"statusCode": 400, "message": "invalid image"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(make_image(64, 64)))
    assert "请求参数有误" in str(info.value) and "invalid image" in str(info.value)


# ---------------------------------------------------------------- 余额


SUBSCRIPTION = {
    "tier": 3, "active": True, "expiresAt": 1795000000, "isGracePeriod": False,
    "trainingStepsLeft": {"fixedTrainingStepsLeft": 9800, "purchasedTrainingSteps": 500},
    "usage": {"isNegative": False, "percent": 63, "purchasedPercent": 0, "timeUntilNextPercent": 6180},
}


def test_balance_success():
    rec = Recorder(httpx.Response(200, json=SUBSCRIPTION))
    info = run(backend(base_url="image.novelai.net/"), rec, method="balance")
    request = rec.requests[0]
    assert request.method == "GET" and str(request.url) == SUB_URL
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert isinstance(info, BalanceInfo)
    expiry = nai.format_expiry(1795000000)
    assert info.items == [
        ("订阅", f"Opus（有效，{expiry} 到期）"),
        ("Anlas", "10300（订阅 9800 + 购买 500）"),
        ("V5 用量条", "剩余 63%，约 103 分钟后恢复 1%"),
    ]
    assert TOKEN not in json.dumps(info.items, ensure_ascii=False) + info.note


def test_balance_info_variants():
    negative = dict(SUBSCRIPTION, usage={"isNegative": True, "percent": -5, "purchasedPercent": 50,
                                         "timeUntilNextPercent": 600})
    assert nai.balance_info(negative).items[2] == (
        "V5 用量条", "已用尽（V5 改扣 Anlas，恢复后重新免费），额外用量 +50%，约 10 分钟后恢复 1%")
    full = dict(SUBSCRIPTION, usage={"isNegative": False, "percent": 104, "timeUntilNextPercent": 0})
    assert nai.balance_info(full).items[2] == ("V5 用量条", "剩余 100%（超过 100% 时暂停恢复）")
    # 没到 100% 时不说"暂停恢复"（timeUntilNextPercent 为 0 只在超过上限时表示暂停）
    zero_wait = dict(SUBSCRIPTION, usage={"isNegative": False, "percent": 63, "timeUntilNextPercent": 0})
    assert nai.balance_info(zero_wait).items[2] == ("V5 用量条", "剩余 63%")
    slow = dict(SUBSCRIPTION, usage={"percent": 10, "timeUntilNextPercent": 3 * 3600})
    assert nai.balance_info(slow).items[2] == ("V5 用量条", "剩余 10%，约 3.0 小时后恢复 1%")
    # 非 Opus 或订阅未激活时不显示用量条
    tablet = {"tier": 1, "active": False, "expiresAt": 0, "usage": {"percent": 50},
              "trainingStepsLeft": {"fixedTrainingStepsLeft": 0, "purchasedTrainingSteps": 120}}
    assert nai.balance_info(tablet).items == [("订阅", "Tablet（未激活）"), ("Anlas", "120（订阅 0 + 购买 120）")]
    # 接口没有返回 Anlas
    items = nai.balance_info({"tier": 3, "active": True}).items
    assert items[0] == ("订阅", "Opus（有效）") and items[1][0] == "Anlas" and items[1][1].startswith("未知")
    assert len(items) == 2


def test_balance_unauthorized_not_retried(sleeps):
    rec = Recorder(httpx.Response(401, json={"statusCode": 401, "message": "Unauthorized"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, method="balance")
    assert info.value.status == 401 and "Token 无效" in str(info.value)
    assert TOKEN not in str(info.value) and len(rec.requests) == 1 and not sleeps


def test_balance_connect_error_retried_once(sleeps):
    rec = Recorder(httpx.ConnectError("refused"), httpx.Response(200, json=SUBSCRIPTION))
    info = run(backend(), rec, method="balance")
    assert info.items[1] == ("Anlas", "10300（订阅 9800 + 购买 500）")
    assert len(rec.requests) == 2 and len(sleeps) == 1

    rec = Recorder(httpx.ConnectError("refused"))
    with pytest.raises(DrawError) as err:
        run(backend(), rec, method="balance")
    assert "无法连接 NovelAI" in str(err.value) and len(rec.requests) == 2


def test_balance_invalid_json():
    rec = Recorder(httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, method="balance")
    assert "订阅信息" in str(info.value)


def test_balance_requires_client():
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=SUBSCRIPTION)))
        try:
            return await backend().balance(Clients(http=pool))
        finally:
            await pool.close()

    with pytest.raises(DrawError) as info:
        asyncio.run(main())
    assert "客户端未初始化" in str(info.value)


# ---------------------------------------------------------------- 配置段落


def load_schema_section() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["novelai"]


def test_schema_section_defaults_match_code():
    section = load_schema_section()
    items = section["items"]
    assert section["type"] == "object"
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field.get("description"), name
        if "options" in field:
            assert field["default"] in field["options"] and len(field.get("labels", field["options"])) == len(
                field["options"])
    assert items["token"]["secret"] is True and items["token"]["default"] == ""
    assert "余额" in items["token"]["hint"]
    assert items["base_url"]["default"] == DEFAULT_BASE_URL
    defaults = NovelAIBackend({})
    assert items["model"]["default"] == defaults.model
    assert "修改图片" in items["model"]["hint"] and "inpainting" in items["model"]["hint"]
    assert {o: items[f"{o}_size"]["default"] for o in DEFAULT_SIZES} == DEFAULT_SIZES
    assert items["steps"]["default"] == NovelAIBackend({"opus_free_guard": False}).steps
    assert items["opus_free_guard"]["default"] is True and defaults.free_guard is True
    assert items["timeout"]["default"] == int(defaults.timeout)
    noise = items["edit_noise"]
    assert noise["type"] == "float" and noise["default"] == defaults.edit_noise == 0.0
    assert noise["slider"]["min"] == 0 and noise["slider"]["max"] == nai.EDIT_MAX_NOISE
    assert noise["hint"] and "description" in noise


def test_schema_defaults_produce_valid_edit_request():
    items = load_schema_section()["items"]
    section = {name: field["default"] for name, field in items.items()}
    section["token"] = TOKEN
    b = NovelAIBackend(section)
    assert b.missing() == "" and b.edit_missing() == "" and b.balance_missing() == ""
    p = b.build_request(edit_req(make_image(1000, 1500), strength=0.6))
    assert (p.width, p.height, p.steps, p.noise, p.strength) == (832, 1216, 28, 0.0, 0.6)
    payload = nai.build_payload(p)
    assert payload["action"] == "img2img" and payload["parameters"]["sampler"] == section["sampler"]
    assert payload["parameters"]["noise_schedule"] == section["noise_schedule"]
