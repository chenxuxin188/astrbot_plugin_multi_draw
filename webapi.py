"""「绘图设置」页面（pages/draw-settings）使用的后端 Web API。

页面通过 AstrBot 的插件页面 bridge 调用这些接口：bridge.apiGet("state") 会被转发到
/{PLUGIN_NAME}/state。AstrBot 4.26 起提供 astrbot.api.web，更早的版本回退到 Quart 的 request。

所有接口都直接返回 dict：成功时是业务数据；失败时是 {"status": "error", "message": ...}，
HTTP 状态码仍为 200——4.24/4.25 的页面 bridge 遇到非 2xx 响应只会显示通用错误，看不到具体原因。
"""

import asyncio
import base64
import time
from pathlib import Path

from astrbot.api import logger

from .backends import (
    BACKEND_ORDER,
    BACKENDS,
    backend_balance_missing,
    backend_edit_missing,
    backend_missing,
    backend_summary,
    create_backend,
)
from .backends.base import (
    SECRET_MASK,
    DrawError,
    DrawRequest,
    InputImage,
    as_bool,
    as_float,
    as_int,
    as_str,
    decode_b64_image,
    image_orientation,
)
from .imaging import normalize_input
from .settings import draft_section, load_schema, masked_config, merge_config
from .usage import estimate_cost, format_cost, period_label, period_start, summarize

try:  # AstrBot >= 4.26
    from astrbot.api.web import request

    _NEW_WEB = True
except ImportError:  # AstrBot <= 4.25（Quart）
    from quart import request  # type: ignore

    _NEW_WEB = False

LOG_PREFIX = "[multi_draw]"
PLUGIN_DIR = Path(__file__).resolve().parent
MAX_TEST_PROMPT_CHARS = 4000
MAX_TEST_IMAGES = 16
MAX_TEST_IMAGE_BYTES = 20 * 1024 * 1024


def _error(message: str) -> dict:
    return {"status": "error", "message": message, "data": {}}


async def _json_body() -> dict:
    try:
        data = await (request.json(default={}) if _NEW_WEB else request.get_json(silent=True))
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


def _query(name: str, default: str = "") -> str:
    try:
        value = (request.query if _NEW_WEB else request.args).get(name)
    except Exception:
        value = None
    return default if value is None else str(value)


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _err_text(e: BaseException) -> str:
    return str(e).strip() or type(e).__name__


class WebApi:
    def __init__(self, plugin, plugin_name: str):
        self.plugin = plugin
        self.plugin_name = plugin_name
        self.prefix = f"/{plugin_name}/"

    def register(self, context) -> None:
        routes = [
            ("state", self.state, ["GET"], "绘图设置：读取配置"),
            ("config", self.save_config, ["POST"], "绘图设置：保存配置"),
            ("test", self.test, ["POST"], "绘图设置：试画"),
            ("models", self.models, ["POST"], "绘图设置：获取模型列表"),
            ("balance", self.balance, ["POST"], "绘图设置：查询余额"),
            ("audit", self.audit, ["GET"], "绘图设置：绘图记录"),
            ("usage", self.usage, ["GET"], "绘图设置：消耗统计"),
            ("images", self.images, ["GET"], "绘图设置：图库列表"),
            ("image", self.image, ["GET"], "绘图设置：读取图片"),
        ]
        for endpoint, handler, methods, desc in routes:
            try:
                context.register_web_api(self.prefix + endpoint, handler, methods, desc)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} 注册页面接口 {endpoint} 失败：{e}")

    def unregister(self, context) -> None:
        """插件停用时移除本插件注册的接口：AstrBot 不会自动移除，停用后旧实例仍会响应请求。"""
        apis = getattr(context, "registered_web_apis", None)
        if isinstance(apis, list):
            apis[:] = [a for a in apis if not str(a[0]).startswith(self.prefix)]

    # ------------------------------------------------------------------ 辅助

    def _schema(self) -> dict:
        return load_schema(PLUGIN_DIR)

    def _backends(self) -> list[dict]:
        out = []
        for key in BACKEND_ORDER:
            backend = self.plugin.backend(key)
            try:
                prompt_mode = backend.prompt_mode
            except Exception:
                prompt_mode = backend.default_prompt_mode
            try:
                edit_prompt_mode = backend.edit_prompt_mode
            except Exception:
                edit_prompt_mode = prompt_mode
            out.append({
                "key": key,
                "label": backend.label,
                "prompt_mode": prompt_mode,
                "edit_prompt_mode": edit_prompt_mode,
                "supports_negative": backend.supports_negative,
                "supports_models": backend.supports_models,
                "max_edit_images": backend.max_edit_images,
                "supports_strength": backend.supports_strength,
                "supports_balance": backend.supports_balance,
                "doc_url": backend.doc_url,
                "billing_url": backend.billing_url,
                # 单个平台的配置出错时只在该平台上显示错误，不影响整个页面
                "missing": backend_missing(backend),
                "edit_missing": backend_edit_missing(backend),
                "balance_missing": backend_balance_missing(backend),
                "summary": backend_summary(backend),
            })
        return out

    def _llm_providers(self) -> list[dict]:
        out = []
        try:
            providers = self.plugin.context.get_all_providers()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} 获取 LLM 提供商失败：{e}")
            return out
        for provider in providers or []:
            try:
                meta = provider.meta()
                modalities = (getattr(provider, "provider_config", None) or {}).get("modalities")
                vision = "image" in modalities if isinstance(modalities, list) and modalities else None
                out.append({"id": meta.id, "name": meta.id, "model": meta.model or "", "vision": vision})
            except Exception:
                continue
        return out

    def _personas(self) -> list[dict]:
        pm = getattr(self.plugin.context, "persona_manager", None)
        try:
            personas = list(getattr(pm, "personas_v3", None) or [])
        except Exception:
            personas = []
        out = [{"name": p.get("name")} for p in personas if isinstance(p, dict) and p.get("name")]
        if not any(p["name"] == "default" for p in out):
            # 与 AstrBot 自带的人格选择器一致：总是可以选择内置的默认人格
            out.insert(0, {"name": "default", "label": "默认人格（AstrBot 内置）"})
        return out

    def _draft_backend(self, payload: dict):
        key = as_str(payload.get("backend"))
        if key not in BACKENDS:
            raise DrawError(f"未知的绘图平台：{key or '（空）'}")
        try:
            section = draft_section(self._schema(), self.plugin.config, key, payload.get("section"))
        except ValueError as e:
            raise DrawError(str(e)) from None
        return create_backend(key, section)

    # ------------------------------------------------------------------ 接口

    async def state(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        schema = self._schema()
        return {
            "version": self._version(),
            "schema": schema,
            "config": masked_config(schema, self.plugin.config),
            "backends": self._backends(),
            "llm_providers": self._llm_providers(),
            "personas": self._personas(),
            "secret_mask": SECRET_MASK,
        }

    async def save_config(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        payload = await _json_body()
        schema = self._schema()
        async with self.plugin.config_lock:
            changed, errors = merge_config(schema, self.plugin.config, payload.get("config"))
            if errors:
                return _error("保存失败：" + "；".join(errors))
            if changed:
                try:
                    await self.plugin.save_config()
                except Exception as e:
                    logger.error(f"{LOG_PREFIX} 保存配置失败：{_err_text(e)}")
                    return _error(f"保存失败：{_err_text(e)}")
                logger.info(f"{LOG_PREFIX} 绘图设置页面修改了配置：{', '.join(changed)}")
                self.plugin.wake_queues()  # 调高并发数时立即放行排队的请求
        return {
            "changed": changed,
            "config": masked_config(schema, self.plugin.config),
            "backends": self._backends(),
        }

    async def test(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        payload = await _json_body()
        prompt = as_str(payload.get("prompt"))[:MAX_TEST_PROMPT_CHARS]
        if not prompt:
            return _error("请先填写试画的提示词")
        try:
            backend = self._draft_backend(payload)
        except DrawError as e:
            return _error(str(e))
        missing = backend_missing(backend, edit=bool(payload.get("images")))
        if missing:
            return _error(missing)
        try:
            images = await self._test_images(payload.get("images"), backend)
        except DrawError as e:
            return _error(str(e))
        orientation = as_str(payload.get("orientation"))
        if orientation not in ("portrait", "landscape", "square"):
            orientation = image_orientation(images[0]) if images else "portrait"
        strength = 0.0
        if images and backend.supports_strength:
            strength = as_float(payload.get("strength"), 0.0, 0.0, 0.99) or as_float(
                self.plugin.section("draw").get("edit_strength"), 0.6, 0.05, 0.99)
        req = DrawRequest(
            prompt=prompt,
            negative=as_str(payload.get("negative"))[:MAX_TEST_PROMPT_CHARS],
            orientation=orientation or "portrait",
            images=images,
            strength=strength,
        )
        started = time.monotonic()
        try:
            async with self.plugin.slot(backend):
                result = await backend.generate(self.plugin.clients, req)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 试画失败（{backend.label}）：{_err_text(e)}")
            return _error(f"试画失败：{_err_text(e)}")
        cost = ""
        usage = result.params.get("usage")
        if isinstance(usage, dict) and as_bool(backend.section.get("track_usage"), True):
            amount, currency, estimated = estimate_cost(usage, backend.section)
            if amount is not None:
                cost = format_cost(amount, currency, estimated)
        return {
            "image": _data_url(result.image, result.mime),
            "mime": result.mime,
            "elapsed": round(time.monotonic() - started, 1),
            "params": result.params,
            "cost": cost,
            "text": result.text[:2000],
        }

    async def _test_images(self, raw, backend) -> list[InputImage]:
        """试画「图片修改」上传的图片（data URL 列表）：校验数量和大小，整理成统一格式。"""
        if not raw:
            return []
        if not isinstance(raw, list):
            raise DrawError("图片格式错误")
        reason = backend_edit_missing(backend)
        if reason:
            raise DrawError(reason)
        if len(raw) > min(MAX_TEST_IMAGES, backend.max_edit_images):
            raise DrawError(f"{backend.label} 修改图片最多使用 {backend.max_edit_images} 张图片")
        images = []
        for index, item in enumerate(raw, 1):
            data = decode_b64_image(str(item or ""), f"第 {index} 张图片")
            if len(data) > MAX_TEST_IMAGE_BYTES:
                raise DrawError(f"第 {index} 张图片超过 20 MB")
            normalized = await asyncio.to_thread(normalize_input, data)
            if normalized is None:
                raise DrawError(f"第 {index} 张图片无法识别")
            images.append(InputImage(*normalized))
        return images

    async def balance(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        payload = await _json_body()
        try:
            backend = self._draft_backend(payload)
        except DrawError as e:
            return _error(str(e))
        reason = backend_balance_missing(backend)
        if reason:
            return _error(reason)
        try:
            info = await backend.balance(self.plugin.clients)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return _error(f"查询余额失败：{_err_text(e)}")
        return {"items": [[str(k), str(v)] for k, v in info.items], "note": info.note}

    async def models(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        payload = await _json_body()
        try:
            backend = self._draft_backend(payload)
            if not backend.supports_models:
                raise DrawError(f"{backend.label} 不支持获取模型列表")
            models = await backend.list_models(self.plugin.clients)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return _error(f"获取模型列表失败：{_err_text(e)}")
        return {"models": models}

    async def audit(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        page = as_int(_query("page", "1"), 1, 1)
        page_size = as_int(_query("page_size", "20"), 20, 1, 100)
        records, page, pages, total = await self.plugin.audit.page(page, page_size)
        return {"records": records, "page": page, "pages": pages, "total": total}

    async def usage(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        days = as_int(_query("days", "30"), 30, 0, 3650)
        backend = _query("backend")
        if backend and backend not in BACKENDS:
            return _error(f"未知的绘图平台：{backend}")
        since = period_start(days)
        records = await self.plugin.audit.records()
        trimmed = await self.plugin.audit.trimmed_until()
        summary = summarize(records, since, None, backend or None, BACKEND_ORDER, trimmed)
        summary.update(days=days, label=period_label(days), since=since.isoformat() if since else "")
        return summary

    async def images(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        limit = as_int(_query("limit", "60"), 60, 1, 500)
        return {"images": await self.plugin.gallery.recent(limit)}

    async def image(self, **_):
        if self.plugin._closing:
            return _error("插件已停用或正在重载，请刷新页面后重试")
        name = _query("name")
        found = await self.plugin.gallery.read(name)
        if found is None:
            return _error("图片不存在或已被清理")
        data, mime = found
        return {"name": name, "image": _data_url(data, mime)}

    @staticmethod
    def _version() -> str:
        try:
            for line in (PLUGIN_DIR / "metadata.yaml").read_text(encoding="utf-8").splitlines():
                if line.startswith("version:"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return ""
