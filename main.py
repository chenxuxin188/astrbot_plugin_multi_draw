import asyncio
import contextlib
import os
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools

from .audit import AuditLog, format_record
from .backends import (
    BACKEND_ORDER,
    BACKENDS,
    backend_balance_missing,
    backend_edit_missing,
    backend_missing,
    backend_summary,
    create_backend,
    resolve_key,
)
from .backends.base import (
    Backend,
    Clients,
    DrawRequest,
    DrawResult,
    HttpPool,
    InputImage,
    as_bool,
    as_float,
    as_int,
    as_str,
    image_orientation,
)
from .backends.novelai import (
    NovelAIBackend,
    NovelAIClient,
    anlas_total,
    format_expiry,
    is_opus_free,
    is_v5_model,
    mask_token,
    mask_url_credentials,
    tier_name,
)
from .backends.novelai import (
    balance_info as novelai_balance_info,
)
from .event_images import (
    ImageCollector,
    collect_event_images,
    count_image_parts,
    load_ref_groups,
    message_image_refs,
)
from .gallery import Gallery
from .imaging import normalize_input, to_jpeg
from .limiter import Limiter
from .prompting import (
    FALLBACK_LINES,
    build_edit_system_prompt,
    build_edit_user_prompt,
    build_natural_system_prompt,
    build_persona_system_prompt,
    build_persona_user_prompt,
    build_tag_edit_system_prompt,
    build_tag_retry_prompt,
    build_tag_system_prompt,
    build_tag_user_prompt,
    choose_tag_result,
    clean_line,
    contains_non_english,
    drop_non_english_tags,
    join_tags,
    normalize_orientation,
    parse_natural_output,
    parse_strength,
    parse_tag_output,
    strip_think,
)
from .usage import apply_usage, format_summary, period_start, summarize
from .webapi import WebApi

PLUGIN_NAME = "astrbot_plugin_multi_draw"
LOG_PREFIX = "[multi_draw]"

PERSONA_LLM_TIMEOUT = 30
VISION_LLM_TIMEOUT = 60
PROMPT_LLM_TIMEOUT = 60
MAX_DESCRIPTION_CHARS = 1000
ANLAS_SETTLE_SECONDS = 1.5
# 看图评价时发给 LLM 的图片长边上限：足够看清画面，又不会让请求过大
EVAL_IMAGE_MAX_EDGE = 1024
# 用户说「把刚才那张图改一下」时，可以修改本会话中多久以内画的图
LAST_IMAGE_TTL = 2 * 3600
LAST_IMAGE_SESSIONS = 100
IMAGE_FETCH_TIMEOUT = 30
# 用户先发图片、再单独发文字要求修改时，可以修改多久以内发的图（只记地址，不下载）
USER_IMAGE_TTL = 30 * 60
USER_IMAGE_ENTRIES = 500
# 这些平台在流水线结束后会关闭回复通道，后台任务发出的消息会被丢弃，需要等绘图完成
INLINE_PLATFORMS = ("webchat", "wecom_ai_bot")
# /draw usage 和 draw_usage 默认统计的天数
DEFAULT_USAGE_DAYS = 30
USAGE_ALL_WORDS = ("all", "全部", "所有")
USAGE_TODAY_WORDS = ("today", "今天", "今日")
# 等不到绘图完成时给事件打上的标记（值为当时的 monotonic 时间）：之后的消息改为通过 context.send_message 主动发送
DETACHED_KEY = "multi_draw_detached"
# 提前结束本轮后，至少等这么久再主动发送：WebUI 对话关闭本轮的回复通道需要一点时间，太早发送的消息会丢失
DETACH_SETTLE_SECONDS = 3.0


def _err_text(e: BaseException) -> str:
    return str(e).strip() or type(e).__name__


def _quiet(task: asyncio.Task) -> asyncio.Task:
    """取走任务的异常，避免任务未被 await 时出现 "Task exception was never retrieved"。

    await 该任务时异常仍会正常抛出。
    """
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return task


def _persona_text(persona) -> str:
    if not persona:
        return ""
    if isinstance(persona, dict):
        text = persona.get("prompt") or persona.get("system_prompt") or ""
    else:
        text = getattr(persona, "system_prompt", None) or getattr(persona, "prompt", None) or ""
    return str(text).strip()


class MultiDrawPlugin(Star):
    """多平台绘图：直接让 Bot 画画；/draw usage 查看消耗；管理员可用 /draw use、/draw balance、/draw audit [页码]"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.http = HttpPool()
        self.nai = NovelAIClient()
        self.clients = Clients(http=self.http, novelai=self.nai)
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.audit = AuditLog(data_dir / "audit.jsonl")
        self.gallery = Gallery(data_dir / "images")
        self.eval_dir = data_dir / "eval"
        self.config_lock = asyncio.Lock()
        self._limiters: dict[str, Limiter] = {}
        # 每个会话最近画好的一张图：用户没有发图时，修改图片默认修改这张
        self._last_images: OrderedDict[str, tuple[InputImage, float]] = OrderedDict()
        # 每个会话中每个用户最近发送的图片地址
        self._user_images: OrderedDict[tuple[str, str], tuple[list[list[str]], float]] = OrderedDict()
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        self.web = WebApi(self, PLUGIN_NAME)
        self.web.register(context)

    async def terminate(self):
        self._closing = True
        self.web.unregister(self.context)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.http.close()
        await self.nai.close()

    # ------------------------------------------------------------------ 配置

    def section(self, name: str) -> dict:
        value = self.config.get(name)
        return value if isinstance(value, dict) else {}

    def backend_key(self) -> str:
        key = as_str(self.section("draw").get("backend")) or "novelai"
        if key not in BACKENDS:
            logger.warning(f"{LOG_PREFIX} 未知的绘图平台 {key}，改用 NovelAI")
            key = "novelai"
        return key

    def backend(self, key: str | None = None) -> Backend:
        key = key or self.backend_key()
        return create_backend(key, self.section(key))

    def edit_backend(self) -> Backend:
        """修改图片使用的平台：配置了「修改图片使用的平台」时用它，否则与绘图相同。"""
        key = as_str(self.section("draw").get("edit_backend"))
        return self.backend(key if key in BACKENDS else None)

    async def save_config(self) -> None:
        """保存插件配置（不会触发插件重载）。调用方需持有 config_lock。"""
        saver = getattr(self.config, "save_config_async", None)  # AstrBot >= 4.27
        if callable(saver):
            await saver()
        else:
            self.config.save_config()

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        raw = self.config.get("admin_ids") or []
        if isinstance(raw, (str, int)):
            raw = re.split(r"[,，\s]+", str(raw))
        admins = {str(x).strip() for x in raw if str(x).strip()}
        return str(event.get_sender_id()).strip() in admins

    # ------------------------------------------------------------------ LLM 工具与指令

    @filter.llm_tool(name="draw_image")
    async def draw_image_tool(self, event: AstrMessageEvent, description: str = "", orientation: str = "",
                              **kwargs):
        """画一张图并直接发送给用户。当用户请你画画、画图、生成图片/插画/头像/壁纸时调用；只是普通聊天时不要调用。插件会自动用你的口吻发送开工提示、图片和评价，所以调用此工具时不要再输出任何回复文字。

        Args:
            description(string): 要画的内容，用自然语言完整描述：角色、外貌、服装、动作、场景、氛围、画风等用户的全部要求，可结合上下文补全
            orientation(string): 可选的画面方向：portrait（竖图）、landscape（横图）或 square（方图），不确定就留空
        """
        # 绘图耗时可能超过 AstrBot 的工具调用超时，因此放到后台任务中进行。
        # 返回 None 会结束本轮 Agent 循环，主 LLM 不会再额外回复一条消息。
        task = self._spawn(self._draw_flow(event, str(description or ""), str(orientation or ""), "llm"))
        if task is not None and event.get_platform_name() in INLINE_PLATFORMS:
            # 在工具调用超时之前尽量等绘图完成；等不及时先结束本轮，之后的消息改为主动发送。
            # asyncio.wait 被取消或超时都不会连带取消绘图任务。
            try:
                done, _ = await asyncio.wait({task}, timeout=self._inline_wait_seconds(event))
            except asyncio.CancelledError:
                event.set_extra(DETACHED_KEY, time.monotonic())
                raise
            if not done:
                event.set_extra(DETACHED_KEY, time.monotonic())
        return None

    async def _prefetch_images(self, event: AstrMessageEvent) -> tuple[list[bytes], int]:
        """读取要修改的图片，返回 (图片数据, 看到的图片数)。

        依次读取本条消息、引用的消息中的图片；都没有时读取该用户最近单独发送的图片。
        读取超时或出错时保留已经读到的图片；消息中有图片时看到的图片数至少为 1，以免改错图片。
        """
        collector = None
        try:
            client = await self.http.get("")
            collector = ImageCollector(client, max(1, self.edit_backend().max_edit_images))
            await asyncio.wait_for(self._collect_images(event, collector), IMAGE_FETCH_TIMEOUT)
            return collector.results, collector.seen
        except asyncio.TimeoutError:
            logger.warning(f"{LOG_PREFIX} 读取消息中的图片超时")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 读取消息中的图片失败：{_err_text(e)}")
        results = collector.results if collector else []
        seen = collector.seen if collector else 0
        if not seen and count_image_parts(event):
            seen = 1
        return results, seen

    async def _collect_images(self, event: AstrMessageEvent, collector: ImageCollector) -> None:
        await collect_event_images(event, self.context, collector)
        if collector.results or collector.seen:
            return
        # 本条消息中没有图片：用户可能先发了图片，再单独发文字要求修改
        umo = event.unified_msg_origin
        cached = self._user_images.get((umo, str(event.get_sender_id())))
        if not cached or time.monotonic() - cached[1] > USER_IMAGE_TTL:
            return
        drawn = self._last_images.get(umo)
        if drawn and drawn[1] > cached[1]:
            return  # 机器人在那之后又画了图：修改最新画的图
        await load_ref_groups(collector, cached[0], "之前发送的消息")

    def _inline_wait_seconds(self, event: AstrMessageEvent) -> float:
        """AstrBot 的工具调用超时减去余量。4.28 起在 agent_runner 设置中，之前在 provider_settings 中。"""
        try:
            cfg = self.context.get_config(umo=event.unified_msg_origin) or {}
        except Exception:
            cfg = {}
        timeout = 0.0
        for path in (("agent_runner", "config", "misc", "tool_call_timeout"), ("provider_settings", "tool_call_timeout")):
            value = cfg
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if value is not None:
                timeout = as_float(value, 0.0)
                break
        return max(5.0, (timeout or 60.0) - 10.0)

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(self, event: AstrMessageEvent, instruction: str = "", orientation: str = "",
                              **kwargs):
        """修改图片并直接发送给用户。当用户发来图片（或引用了一张图片）并要求修改、换背景、换衣服、换画风、加减元素、重绘时调用；用户说「把刚才那张图改一下」这类话时也调用，会修改你上一张画的图。插件会自动用你的口吻发送开工提示、修改后的图片和评价，所以调用此工具时不要再输出任何回复文字。

        Args:
            instruction(string): 要怎么修改，用自然语言完整描述用户的要求：改什么、哪些地方保持不变
            orientation(string): 可选，只有用户要求改变画面方向时才填写：portrait（竖图）、landscape（横图）或 square（方图），否则留空以保持原图比例
        """
        started = time.monotonic()
        # 必须现在就读取图片：AstrBot 4.26-4.28 会在本轮事件结束后删除图片的临时文件，QQ 图片链接也会过期
        prefetched = await self._prefetch_images(event)
        task = self._spawn(self._draw_flow(event, str(instruction or ""), str(orientation or ""), "llm",
                                           edit=True, prefetched=prefetched))
        if task is not None and event.get_platform_name() in INLINE_PLATFORMS:
            wait = max(1.0, self._inline_wait_seconds(event) - (time.monotonic() - started))
            try:
                done, _ = await asyncio.wait({task}, timeout=wait)
            except asyncio.CancelledError:
                event.set_extra(DETACHED_KEY, time.monotonic())
                raise
            if not done:
                event.set_extra(DETACHED_KEY, time.monotonic())
        return None

    @filter.llm_tool(name="draw_usage")
    async def draw_usage_tool(self, event: AstrMessageEvent, days: int = DEFAULT_USAGE_DAYS, platform: str = "",
                              **kwargs):
        """查询画图的消耗统计：画了多少张、用了多少 token 或 Anlas、大约花了多少钱；管理员还会看到各绘图平台的余额。当用户问画图花了多少钱、消耗了多少、今天画了几张、还剩多少余额或额度时调用。查到的结果请用你的口吻转述给用户。

        Args:
            days(number): 统计最近几天：1 表示今天，7 表示最近 7 天，0 表示全部记录；用户没说时填 30
            platform(string): 可选，只统计某个绘图平台，例如 novelai、openai、gemini；不确定就留空，统计全部平台
        """
        days = as_int(days, DEFAULT_USAGE_DAYS, 0, 3650)
        key = None
        if as_str(platform):
            key = resolve_key(as_str(platform))
            if not key:
                return f"没有名为「{platform}」的绘图平台，可选：{'、'.join(BACKEND_ORDER)}"
        admin = self._is_admin(event)
        text = await self.usage_text(days, None if admin else str(event.get_sender_id()), key)
        if not admin:
            return text + "\n（只统计了该用户自己的绘图；各平台的余额只有管理员可以查询）"
        keys = [key] if key else [k for k in BACKEND_ORDER if not backend_missing(self.backend(k))]
        keys = [k for k in keys if not backend_balance_missing(self.backend(k))]
        if keys:
            results = await asyncio.gather(*(self.query_balance(self.backend(k)) for k in keys))
            text += "\n各平台余额：\n" + "\n".join(
                f"【{BACKENDS[k].label}】{r}" for k, r in zip(keys, results, strict=True))
        return text

    async def usage_text(self, days: int, user_id: str | None = None, backend: str | None = None) -> str:
        """最近 days 天（0 为全部）的绘图消耗统计文字。user_id 不为空时只统计该用户。"""
        records = await self.audit.records()
        summary = summarize(records, period_start(days), user_id, backend, BACKEND_ORDER,
                            await self.audit.trimmed_until())
        whose = "你的" if user_id is not None else ""
        if backend:
            whose += f"{BACKENDS[backend].label} "
        return format_summary(summary, days, whose)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def remember_user_images(self, event: AstrMessageEvent):
        """记下用户最近发送的图片地址（不下载），用户之后单独发文字要求修改时使用。"""
        try:
            groups = message_image_refs(event)
        except Exception:
            return
        if not groups:
            return
        key = (event.unified_msg_origin, str(event.get_sender_id()))
        self._user_images[key] = (groups, time.monotonic())
        self._user_images.move_to_end(key)
        while len(self._user_images) > USER_IMAGE_ENTRIES:
            self._user_images.popitem(last=False)

    @filter.command_group("draw")
    def draw(self):
        """多平台绘图"""

    @draw.command("status")
    async def draw_status(self, event: AstrMessageEvent):
        """查看当前绘图平台和各平台的配置情况"""
        current = self.backend_key()
        lines = [f"当前绘图平台：{BACKENDS[current].label}（{backend_summary(self.backend())}）", "各平台："]
        for key in BACKEND_ORDER:
            missing = backend_missing(self.backend(key))
            mark = "▶" if key == current else ("✅" if not missing else "⚪")
            lines.append(f"{mark} {key}：{BACKENDS[key].label}" + (f"（{missing}）" if missing else ""))
        lines.append("管理员可用 /draw use <平台> 切换，例如 /draw use openai")
        yield event.plain_result("\n".join(lines)).use_t2i(False)

    @draw.command("use")
    async def draw_use(self, event: AstrMessageEvent, name: str = ""):
        """切换绘图平台（管理员）：/draw use <平台>"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足：仅插件配置中的管理员可以使用该指令。")
            return
        key = resolve_key(name)
        if not key:
            names = "、".join(BACKEND_ORDER)
            yield event.plain_result(f"请指定平台：{names}，例如 /draw use openai")
            return
        missing = backend_missing(self.backend(key))
        try:
            async with self.config_lock:
                draw = self.config.get("draw")
                if not isinstance(draw, dict):
                    draw = {}
                    self.config["draw"] = draw
                draw["backend"] = key
                await self.save_config()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 保存配置失败：{_err_text(e)}")
            yield event.plain_result(f"已切换，但保存配置失败（重启后会恢复原设置）：{_err_text(e)}")
            return
        text = f"已切换到 {BACKENDS[key].label}（{backend_summary(self.backend(key))}）。"
        if missing:
            text += f"\n注意：{missing}，在补全配置之前无法画画。"
        yield event.plain_result(text).use_t2i(False)

    @draw.command("account")
    async def draw_account(self, event: AstrMessageEvent):
        """查询 NovelAI 账户与 Anlas 余额（管理员）"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足：仅插件配置中的管理员可以使用该指令。")
            return
        backend = self.backend("novelai")
        if backend.missing():
            yield event.plain_result("尚未配置 NovelAI Token。")
            return
        try:
            data = await backend.subscription(self.clients)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 查询订阅失败：{_err_text(e)}")
            yield event.plain_result(f"查询失败：{_err_text(e)}")
            return
        yield event.plain_result(self._format_account(data, backend)).use_t2i(False)

    @draw.command("balance")
    async def draw_balance(self, event: AstrMessageEvent, name: str = ""):
        """查询绘图平台的余额（管理员）：/draw balance [平台]，不填平台时查询所有已配置的平台"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足：仅插件配置中的管理员可以使用该指令。")
            return
        if name:
            key = resolve_key(name)
            if not key:
                yield event.plain_result(f"请指定平台：{'、'.join(BACKEND_ORDER)}，例如 /draw balance openai")
                return
            keys = [key]
        else:
            keys = [k for k in BACKEND_ORDER if not backend_missing(self.backend(k))]
            if not keys:
                yield event.plain_result("还没有配置任何绘图平台。")
                return
        results = await asyncio.gather(*(self.query_balance(self.backend(k)) for k in keys))
        lines = ["绘图平台余额"]
        for key, text in zip(keys, results, strict=True):
            lines.append(f"【{BACKENDS[key].label}】{text}")
        yield event.plain_result("\n".join(lines)).use_t2i(False)

    async def query_balance(self, backend: Backend) -> str:
        """查询一个平台的余额，返回一段可直接展示的文字。"""
        reason = backend_balance_missing(backend)
        if reason:
            return reason
        try:
            info = await backend.balance(self.clients)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 查询 {backend.label} 余额失败：{_err_text(e)}")
            return f"查询失败：{_err_text(e)}"
        text = "；".join(f"{k}：{v}" for k, v in info.items) or "没有返回余额信息"
        return f"{text}（{info.note}）" if info.note else text

    @draw.command("usage")
    async def draw_usage(self, event: AstrMessageEvent, first: str = "", second: str = ""):
        """查看绘图消耗：/draw usage [天数|today|all] [平台]，默认最近 30 天；管理员看到所有人的，其他人只看到自己的"""
        days, key = DEFAULT_USAGE_DAYS, None
        for word in (as_str(first), as_str(second)):
            if not word:
                continue
            lower = word.lower()
            if lower in USAGE_ALL_WORDS:
                days = 0
            elif lower in USAGE_TODAY_WORDS:
                days = 1
            elif re.fullmatch(r"\d{1,4}", lower):
                days = int(lower)
            elif resolve_key(word):
                key = resolve_key(word)
            else:
                yield event.plain_result(
                    f"无法识别「{word}」。用法：/draw usage [天数|today|all] [平台]，例如 /draw usage 7 openai"
                )
                return
        uid = None if self._is_admin(event) else str(event.get_sender_id())
        yield event.plain_result(await self.usage_text(days, uid, key)).use_t2i(False)

    @draw.command("audit")
    async def draw_audit(self, event: AstrMessageEvent, page: int = 1):
        """查看绘图记录（管理员）：/draw audit [页码]"""
        if not self._is_admin(event):
            yield event.plain_result("权限不足：仅插件配置中的管理员可以使用该指令。")
            return
        page_size = as_int(self.section("audit").get("page_size"), 10, 1, 50)
        records, page, pages, total = await self.audit.page(as_int(page, 1, 1), page_size)
        if not total:
            yield event.plain_result("暂无绘图记录。")
            return
        lines = [f"绘图记录（第 {page}/{pages} 页，共 {total} 条，从新到旧）"]
        lines += [format_record(r) for r in records]
        if page < pages:
            lines.append(f"查看下一页：/draw audit {page + 1}")
        yield event.plain_result("\n".join(lines)).use_t2i(False)

    # ------------------------------------------------------------------ 绘图流程

    def _spawn(self, coro) -> asyncio.Task | None:
        if self._closing:
            coro.close()
            return None
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"{LOG_PREFIX} 绘图任务异常：{task.exception()!r}")

    def slot_limit(self, key: str) -> int:
        fixed = getattr(BACKENDS.get(key), "max_concurrency", None)
        return fixed or as_int(self.section("draw").get("max_concurrency"), 2, 1, 8)

    def _limiter(self, key: str) -> Limiter:
        limiter = self._limiters.get(key)
        if limiter is None:
            limiter = self._limiters[key] = Limiter(lambda: self.slot_limit(key))
        return limiter

    def wake_queues(self) -> None:
        """修改并发数后调用：按新的上限立即放行排队的请求。"""
        for limiter in self._limiters.values():
            limiter.dispatch()

    @contextlib.asynccontextmanager
    async def slot(self, backend: Backend):
        """按平台限制同时进行的请求数，超出的请求按先后顺序排队。"""
        limiter = self._limiter(backend.key)
        await limiter.acquire()
        try:
            yield
        finally:
            limiter.release()

    def _queued_ahead(self, backend: Backend) -> int:
        return self._limiter(backend.key).queued_ahead()

    async def _draw_flow(self, event: AstrMessageEvent, description: str, orientation: str, source: str,
                         edit: bool = False, prefetched: tuple[list[bytes], int] | None = None):
        description = description.strip()[:MAX_DESCRIPTION_CHARS]
        if not description:
            ask = "想把图片改成什么样呢？把修改要求告诉我吧～" if edit else "想让我画什么呢？把想画的内容告诉我吧～"
            await self._send_text(event, ask)
            return
        backend = self.edit_backend() if edit else self.backend()
        missing = backend_missing(backend, edit=edit)
        if missing:
            await self._send_text(event, f"{missing}，暂时没法画画哦。")
            return
        images: list[InputImage] = []
        if edit:
            reason = backend_edit_missing(backend)
            if reason:
                await self._send_text(event, f"{reason}，没法修改图片哦。管理员可以在「修改图片使用的平台」中换一个支持的平台。")
                return
            images, seen = await self._edit_inputs(event, backend.max_edit_images, prefetched)
            if not images:
                await self._send_text(event, "消息里的图片读取失败了（QQ 的图片链接可能已经过期），请重新发送图片再试一次～" if seen
                                      else "没有找到要修改的图片～请把图片和修改要求一起发给我，或者引用那张图片再说要怎么改。")
                return
        uid = str(event.get_sender_id())
        if uid in self._inflight:
            await self._send_text(event, "你的上一张画还没完成，请稍等一下～")
            return
        self._inflight.add(uid)
        try:
            await self._draw(event, description, orientation, source, backend, images)
        finally:
            self._inflight.discard(uid)

    async def _edit_inputs(self, event: AstrMessageEvent, limit: int,
                           prefetched: tuple[list[bytes], int] | None) -> tuple[list[InputImage], int]:
        """修改图片的输入：本条消息或引用的消息中的图片；消息中没有图片时使用本会话最近画的一张图。

        返回 (图片, 消息中看到的图片数)。消息中有图片却读不出来时不改用最近的图，以免改错图片。
        """
        raw, seen = prefetched or ([], 0)
        images = []
        for data in raw:
            normalized = await asyncio.to_thread(normalize_input, data)
            if normalized is None:
                logger.warning(f"{LOG_PREFIX} 跳过无法识别的图片（{len(data)} 字节）")
                continue
            images.append(InputImage(*normalized))
        if images:
            if len(images) > limit:
                logger.info(f"{LOG_PREFIX} 消息中有 {len(images)} 张图片，只使用前 {limit} 张")
            return images[:limit], seen
        if seen:
            return [], seen
        cached = self._last_images.get(event.unified_msg_origin)
        if cached and time.monotonic() - cached[1] < LAST_IMAGE_TTL:
            return [cached[0]], 0
        return [], 0

    async def _remember_image(self, event: AstrMessageEvent, result: DrawResult) -> None:
        normalized = await asyncio.to_thread(normalize_input, result.image)
        if normalized is None:
            return
        umo = event.unified_msg_origin
        self._last_images[umo] = (InputImage(*normalized), time.monotonic())
        self._last_images.move_to_end(umo)
        while len(self._last_images) > LAST_IMAGE_SESSIONS:
            self._last_images.popitem(last=False)

    async def _draw(self, event: AstrMessageEvent, description: str, orientation: str, source: str,
                    backend: Backend, images: list[InputImage] | None = None):
        images = images or []
        started = time.monotonic()
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "user_id": str(event.get_sender_id()),
            "user_name": as_str(event.get_sender_name()),
            "group_id": as_str(event.get_group_id()),
            "umo": event.unified_msg_origin,
            "backend": backend.key,
            "backend_label": backend.label,
            "request": description,
            "status": "failed",
        }
        if images:
            record.update(mode="edit", input_images=len(images))
        with contextlib.suppress(Exception):
            record["model"] = backend.model  # 生成前就失败（如 LLM 不可用）时记录中也有模型
        # 人格台词中的需求描述：修改图片时说明是在改对方的图
        persona_desc = f"修改对方发来的图片：{description}" if images else description
        stage = "prompt"
        persona = ""
        prompt_task = None
        req = None
        result = None
        try:
            persona = await self._persona_prompt(event)
            prompt_task = _quiet(asyncio.create_task(
                self._make_prompt(event, description, orientation,
                                  backend.edit_prompt_mode if images else backend.prompt_mode, images)
            ))
            start_line = await self._persona_line(event, persona, "start", description=persona_desc)
            queued = self._queued_ahead(backend)
            if queued:
                start_line += f"\n（前面还有 {queued} 张在排队）"
            await self._send_text(event, start_line)

            generated = await prompt_task
            req = self._build_request(backend, generated, orientation, images)
            logger.info(f"{LOG_PREFIX} 发送给 {backend.label} 的 prompt：{req.prompt}")
            if req.negative:
                logger.info(f"{LOG_PREFIX} 发送给 {backend.label} 的负面 prompt：{req.negative}")
            record.update(prompt=req.prompt, negative=req.negative, orientation=req.orientation)

            stage = "generate"
            send_task, result = await self._generate(backend, req, record, event)

            stage = "send"
            await send_task
            record["status"] = "success"
        except asyncio.CancelledError:
            record.update(stage=stage, error="任务被取消（插件被停用或重载）")
            if stage != "send":
                # terminate() 只取消一次，这里还能发出一条简短提示；限时避免拖慢重载
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(
                        self._send_text(event, "绘图被中断了（插件正在重载或关闭），请稍后再试～"), 5
                    )
            raise
        except Exception as e:
            error = _err_text(e)
            record.update(stage=stage, error=error[:500])
            logger.warning(f"{LOG_PREFIX} 绘图失败（{backend.label}，{stage}）：{error}")
            await self._send_text(event, f"绘图失败：{error}")
            line = await self._persona_line(event, persona, "fail", description=persona_desc, error=error)
            await self._send_text(event, line)
        else:
            await self._remember_image(event, result)
            comment = await self._persona_line(
                event, persona, "success", image=result, description=persona_desc, prompt=req.prompt
            )
            await self._send_text(event, comment)
        finally:
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()
            record["duration"] = round(time.monotonic() - started, 1)
            await self._save_record(record)

    def _build_request(self, backend: Backend, generated: dict, orientation: str,
                       images: list[InputImage] | None = None) -> DrawRequest:
        images = images or []
        explicit = normalize_orientation(orientation)
        default = normalize_orientation(self.section("draw").get("default_orientation"))
        if images:
            # 修改图片时保持原图的方向；只有用户要求时（edit_image 的 orientation 参数）才改变
            orient = explicit or image_orientation(images[0]) or generated["orientation"] or default or "portrait"
        else:
            orient = explicit or generated["orientation"] or default or "portrait"
        mode = backend.edit_prompt_mode if images else backend.prompt_mode
        positive = as_str(backend.section.get("required_positive"))
        negative = as_str(backend.section.get("required_negative"))
        if images and mode != "tags":
            positive = negative = ""  # 自然语言修改指令后面不追加质量标签
        if mode == "tags":
            # LLM 生成的标签已在 _make_prompt 中过滤，这里只过滤配置里填写的必定包含的提示词
            positive, dropped = drop_non_english_tags(positive)
            negative, dropped_negative = drop_non_english_tags(negative)
            if dropped or dropped_negative:
                logger.warning(f"{LOG_PREFIX} 配置的提示词中含非英文标签，已忽略：{dropped + dropped_negative}")
        if isinstance(backend, NovelAIBackend) and is_v5_model(backend.model) and backend.free_guard:
            logger.warning(f"{LOG_PREFIX} {backend.model} 是 V5 模型，不受免费参数保护：会消耗 Opus 用量条，用尽后扣 Anlas")
        strength = 0.0
        if images and backend.supports_strength:
            strength = generated.get("strength") or as_float(self.section("draw").get("edit_strength"), 0.6, 0.05, 0.99)
        return DrawRequest(
            # 质量词按官方做法追加在末尾，必须的负面词放在最前
            prompt=join_tags(generated["prompt"], positive),
            negative=join_tags(negative, generated["negative"]),
            orientation=orient,
            images=images,
            strength=strength,
        )

    async def _generate(self, backend: Backend, req: DrawRequest, record: dict,
                        event: AstrMessageEvent) -> tuple[asyncio.Task, DrawResult]:
        """按平台排队生成图片，返回 (发送图片的任务, 生成结果)。

        使用 NovelAI 且开启 Anlas 记录时，在占用生成名额期间查询前后余额，差值只属于本次请求。
        其他平台按各自的「记录用量」设置记录平台返回的用量，并按配置的价格估算费用。
        """
        track = isinstance(backend, NovelAIBackend) and as_bool(self.section("novelai").get("track_anlas"), True)
        # 先记下预计使用的参数，生成失败时记录中也有模型和尺寸
        with contextlib.suppress(Exception):
            record.update(backend.plan(req))
        async with self.slot(backend):
            before = await self._anlas(backend) if track else None
            try:
                result = await backend.generate(self.clients, req)
            except Exception:
                if before is not None:  # 失败的请求也可能已经扣费
                    await self._record_anlas(backend, record, before)
                raise
            record.update(result.params)
            apply_usage(record, backend.section)
            if result.text:
                logger.info(f"{LOG_PREFIX} {backend.label} 返回的文字：{result.text[:500]}")
            # 先把图片发出去，等待扣费结算、保存图片与上传同时进行
            image = await self._image_for_sending(result)
            send_task = _quiet(asyncio.create_task(self._deliver(event, event.chain_result([Image.fromBytes(image)]))))
            try:
                if before is not None:
                    await self._record_anlas(backend, record, before)
                await self._save_image(backend, result, record)
            except asyncio.CancelledError:
                send_task.cancel()
                raise
            return send_task, result

    async def _image_for_sending(self, result: DrawResult) -> bytes:
        """超过设置大小的图片转成 JPEG 再发送：QQ 等平台以 base64 内联发送图片，过大时可能发送失败。"""
        limit_mb = as_int(self.section("draw").get("compress_over_mb"), 8, 0)
        if not limit_mb or len(result.image) <= limit_mb * 1024 * 1024:
            return result.image
        compressed = await asyncio.to_thread(to_jpeg, result.image, 0, 90)
        if compressed and len(compressed) < len(result.image):
            logger.info(f"{LOG_PREFIX} 图片 {len(result.image) // 1024} KB 超过 {limit_mb} MB，"
                        f"已压缩为 JPEG {len(compressed) // 1024} KB 后发送")
            return compressed
        return result.image

    async def _save_image(self, backend: Backend, result: DrawResult, record: dict) -> None:
        draw = self.section("draw")
        if not as_bool(draw.get("save_images"), True):
            return
        try:
            path = await self.gallery.save(
                result.image, result.mime, backend.key, as_int(draw.get("keep_images"), 200, 0)
            )
            record["image"] = f"{path.parent.name}/{path.name}"
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 保存图片失败：{_err_text(e)}")

    async def _record_anlas(self, backend: NovelAIBackend, record: dict, before: int) -> None:
        await asyncio.sleep(ANLAS_SETTLE_SECONDS)
        after = await self._anlas(backend)
        record.update(anlas_before=before, anlas_after=after)
        if after is not None:
            record["anlas_cost"] = before - after

    async def _anlas(self, backend: NovelAIBackend) -> int | None:
        try:
            data = await backend.subscription(self.clients)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} 查询 Anlas 失败：{_err_text(e)}")
            return None
        return anlas_total(data)

    async def _save_record(self, record: dict) -> None:
        self.audit.max_records = as_int(self.section("audit").get("max_records"), 2000, 10)
        try:
            await self.audit.append(record)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 写入绘图记录失败：{_err_text(e)}")

    async def _deliver(self, event: AstrMessageEvent, chain) -> None:
        """发送消息；本轮对话已提前结束（见 draw_image_tool）时改为主动发送。"""
        detached_at = event.get_extra(DETACHED_KEY)
        if not detached_at:
            await event.send(chain)
            return
        if isinstance(detached_at, (int, float)):
            wait = DETACH_SETTLE_SECONDS - (time.monotonic() - detached_at)
            if wait > 0:
                await asyncio.sleep(wait)
        await self.context.send_message(event.unified_msg_origin, chain)

    async def _send_text(self, event: AstrMessageEvent, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            await self._deliver(event, event.plain_result(text))
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 发送消息失败：{_err_text(e)}")

    # ------------------------------------------------------------------ LLM

    async def _llm(self, event: AstrMessageEvent, provider_id: str, system_prompt: str, prompt: str,
                   timeout: float, image_urls: list[str] | None = None) -> str:
        """调用 LLM；配置的提供商不可用时回退到当前会话的提供商。"""
        candidates = [provider_id] if provider_id else []
        try:
            session_pid = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} 获取当前会话模型失败：{_err_text(e)}")
            session_pid = ""
        if session_pid and session_pid not in candidates:
            candidates.append(session_pid)
        if image_urls:
            # 跳过明确声明不支持图片输入的模型（AstrBot 的 llm_generate 不会替我们过滤）
            candidates = [pid for pid in candidates if self._supports_image(pid) is not False]
            if not candidates:
                raise RuntimeError("可用的模型都不支持图片输入")
        if not candidates:
            raise RuntimeError("没有可用的 LLM 提供商")
        kwargs = {"image_urls": image_urls} if image_urls else {}
        last_error: Exception | None = None
        for pid in candidates:
            try:
                resp = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=pid, prompt=prompt, system_prompt=system_prompt, **kwargs
                    ),
                    timeout,
                )
                if getattr(resp, "role", "") == "err":
                    raise RuntimeError(resp.completion_text or "LLM 返回错误")
                text = strip_think(resp.completion_text or "")
                if text:
                    return text
                raise RuntimeError("LLM 返回了空内容")
            except Exception as e:
                last_error = e
                logger.warning(f"{LOG_PREFIX} LLM({pid}) 调用失败：{_err_text(e)}")
        raise RuntimeError(f"LLM 调用失败：{_err_text(last_error)}")

    def _supports_image(self, provider_id: str) -> bool | None:
        """按提供商配置的模态判断是否支持图片输入；未配置时返回 None（不确定，直接尝试）。"""
        try:
            provider = self.context.get_provider_by_id(provider_id)
        except Exception:
            return None
        modalities = (getattr(provider, "provider_config", None) or {}).get("modalities")
        if isinstance(modalities, list) and modalities:
            return "image" in modalities
        return None

    async def _prompt_llm(self, event: AstrMessageEvent, provider_id: str, system_prompt: str, description: str,
                          orientation: str, images: list[InputImage]) -> str:
        """生成提示词。修改图片时尽量把原图一起发给 LLM（需要支持图片输入），失败时只发文字。"""
        if not images:
            return await self._llm(event, provider_id, system_prompt, build_tag_user_prompt(description, orientation),
                                   PROMPT_LLM_TIMEOUT)
        paths = []
        try:
            for image in images:
                paths.append(await asyncio.to_thread(self._write_eval_image, DrawResult(image.data, image.mime)))
            return await self._llm(
                event, provider_id, system_prompt,
                build_edit_user_prompt(description, orientation, len(images), True), PROMPT_LLM_TIMEOUT,
                image_urls=paths,
            )
        except Exception as e:
            logger.info(f"{LOG_PREFIX} 带原图生成修改提示词失败，改为只发文字：{_err_text(e)}")
        finally:
            for path in paths:
                with contextlib.suppress(OSError):
                    os.remove(path)
        return await self._llm(event, provider_id, system_prompt,
                               build_edit_user_prompt(description, orientation, len(images), False), PROMPT_LLM_TIMEOUT)

    async def _make_prompt(self, event: AstrMessageEvent, description: str, orientation: str, mode: str,
                           images: list[InputImage] | None = None) -> dict:
        """用 LLM 把绘图需求转换成提示词：tags 模式生成英文标签，natural 模式生成自然语言描述；
        修改图片时 natural 模式生成修改指令，tags 模式生成结果图的标签和重绘幅度。"""
        images = images or []
        pc = self.section("prompt")
        sfw = as_bool(pc.get("sfw_only"), True)
        extra = as_str(pc.get("extra_instruction"))
        provider_id = as_str(pc.get("provider_id"))
        orientation = normalize_orientation(orientation)
        if mode != "tags":
            if images:
                system_prompt = build_edit_system_prompt(as_str(pc.get("edit_system_prompt")), sfw=sfw, extra=extra)
            else:
                system_prompt = build_natural_system_prompt(as_str(pc.get("natural_system_prompt")), sfw=sfw,
                                                            extra=extra)
            raw = await self._prompt_llm(event, provider_id, system_prompt, description, orientation, images)
            result = parse_natural_output(raw)
            if not result["prompt"]:
                raise RuntimeError("LLM 没有生成有效的绘图提示词")
            result["strength"] = parse_strength(raw)
            return result

        build = build_tag_edit_system_prompt if images else build_tag_system_prompt
        system_prompt = build(as_str(pc.get("tag_system_prompt")), sfw=sfw, extra=extra)
        raw = await self._prompt_llm(event, provider_id, system_prompt, description, orientation, images)
        strength = parse_strength(raw)
        tags = parse_tag_output(raw)
        # 标签模式的绘图模型只理解英文：结果里混有中日韩文字时让 LLM 重写一次
        if contains_non_english(tags["prompt"]) or contains_non_english(tags["negative"]):
            logger.info(f"{LOG_PREFIX} 提示词含非英文内容，要求 LLM 改写为英文：{tags['prompt'][:200]}")
            try:
                raw = await self._llm(
                    event,
                    provider_id,
                    system_prompt,
                    build_tag_retry_prompt(description, orientation, raw),
                    PROMPT_LLM_TIMEOUT,
                )
                chosen = choose_tag_result(tags, parse_tag_output(raw))
                if chosen is tags:
                    logger.warning(f"{LOG_PREFIX} 改写结果不如第一次，沿用第一次的结果并去掉非英文标签")
                tags = chosen
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} 改写英文提示词失败，将去掉非英文标签：{_err_text(e)}")
        tags["prompt"], dropped = drop_non_english_tags(tags["prompt"])
        tags["negative"], dropped_negative = drop_non_english_tags(tags["negative"])
        if dropped or dropped_negative:
            logger.warning(f"{LOG_PREFIX} 已去掉无法使用的非英文标签：{dropped + dropped_negative}")
        if not tags["prompt"]:
            raise RuntimeError("LLM 没有生成有效的英文绘图提示词")
        tags["strength"] = strength
        return tags

    async def _persona_line(self, event: AstrMessageEvent, persona: str, kind: str,
                            image: DrawResult | None = None, **info) -> str:
        """按 Bot 人格生成一句台词；失败时使用内置台词。

        生成评价且开启了「评价时附上生成的图片」时，先带图调用看图模型，失败再改为只发文字。
        """
        pc = self.section("persona")
        max_len = as_int(pc.get("max_length"), 40, 10, 200)
        system_prompt = build_persona_system_prompt(persona, max_len)
        template = as_str(pc.get(f"{kind}_prompt"))
        provider_id = as_str(pc.get("provider_id"))
        if image is not None and kind == "success" and as_bool(pc.get("comment_with_image"), False):
            path = None
            try:
                path = await asyncio.to_thread(self._write_eval_image, image)
                text = await self._llm(
                    event,
                    as_str(pc.get("vision_provider_id")) or provider_id,
                    system_prompt,
                    build_persona_user_prompt(kind, template, with_image=True, **info),
                    VISION_LLM_TIMEOUT,
                    image_urls=[path],
                )
                text = clean_line(text, max_len)
                if text:
                    return text
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} 看图评价失败，改为只根据提示词评价：{_err_text(e)}")
            finally:
                if path:
                    with contextlib.suppress(OSError):
                        os.remove(path)
        try:
            text = await self._llm(
                event,
                provider_id,
                system_prompt,
                build_persona_user_prompt(kind, template, **info),
                PERSONA_LLM_TIMEOUT,
            )
            text = clean_line(text, max_len)
            if text:
                return text
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 生成人格台词失败，使用默认台词：{_err_text(e)}")
        return FALLBACK_LINES[kind]

    def _write_eval_image(self, image: DrawResult) -> str:
        """把生成的图片缩小并转成 JPEG，写入临时文件，返回绝对路径。

        llm_generate 的 image_urls 在各版本 AstrBot 中都支持普通文件路径，且旧版本会把文件一律标记为
        image/jpeg，所以统一转成 JPEG；缩小尺寸可以减少看图模型的请求大小和费用。
        """
        data = to_jpeg(image.image, EVAL_IMAGE_MAX_EDGE, 85)
        ext = "jpg"
        if data is None:  # Pillow 不可用时直接使用原图
            data, ext = image.image, {"image/png": "png", "image/webp": "webp"}.get(image.mime, "jpg")
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        path = (self.eval_dir / f"eval_{uuid.uuid4().hex}.{ext}").resolve()
        path.write_bytes(data)
        return str(path)

    async def _persona_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前会话实际使用的人格设定（system prompt）。"""
        pm = getattr(self.context, "persona_manager", None)
        if pm is None:
            return ""
        umo = event.unified_msg_origin
        try:
            override = as_str(self.section("persona").get("persona_id"))
            if override == "[%None]":  # 明确选择不使用人格
                return ""
            if override:
                found = next((p for p in pm.personas_v3 if p.get("name") == override), None)
                if found is None and override == "default":
                    # 人格选择器中的「默认人格」：已有自定义人格时它不在 personas_v3 中
                    found = await self._builtin_default_persona(pm)
                if found:
                    return _persona_text(found)
                logger.warning(f"{LOG_PREFIX} 未找到配置的人格 {override}，改用会话人格")

            conv_pid = await self._conversation_persona_id(event)
            if hasattr(pm, "resolve_selected_persona"):
                provider_settings = (self.context.get_config(umo=umo) or {}).get("provider_settings") or {}
                selected, persona, _, _ = await pm.resolve_selected_persona(
                    umo=umo,
                    conversation_persona_id=conv_pid,
                    platform_name=event.get_platform_name(),
                    provider_settings=provider_settings,
                )
            else:
                selected, persona = conv_pid, None
                if conv_pid and conv_pid != "[%None]":
                    persona = next((p for p in pm.personas_v3 if p.get("name") == conv_pid), None)
            if selected == "[%None]":  # 会话中明确关闭了人格
                return ""
            if persona is None:
                persona = await pm.get_default_persona_v3(umo=umo)
            return _persona_text(persona)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 获取人格失败：{_err_text(e)}")
            return ""

    @staticmethod
    async def _builtin_default_persona(pm):
        """AstrBot 内置的默认人格（id 为 default）。"""
        getter = getattr(pm, "get_persona_v3_by_id", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                found = getter("default")
                if asyncio.iscoroutine(found):
                    found = await found
                if found:
                    return found
        try:
            from astrbot.core.persona_mgr import DEFAULT_PERSONALITY
        except Exception:
            return None
        return DEFAULT_PERSONALITY

    async def _conversation_persona_id(self, event: AstrMessageEvent) -> str | None:
        try:
            req = event.get_extra("provider_request")
            conv = getattr(req, "conversation", None) if req is not None else None
            if conv is None:
                cm = self.context.conversation_manager
                cid = await cm.get_curr_conversation_id(event.unified_msg_origin)
                conv = await cm.get_conversation(event.unified_msg_origin, cid) if cid else None
            return getattr(conv, "persona_id", None) if conv is not None else None
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} 获取会话人格失败：{_err_text(e)}")
            return None

    # ------------------------------------------------------------------ 账户

    def _format_account(self, data: dict, backend: NovelAIBackend) -> str:
        tier = data.get("tier")
        steps = data.get("trainingStepsLeft") if isinstance(data.get("trainingStepsLeft"), dict) else {}
        expiry = format_expiry(data.get("expiresAt"))
        status = "有效" if data.get("active") else "未激活"
        if expiry:
            status += f"，到期 {expiry}"
        total = anlas_total(data)
        proxy = backend.proxy
        lines = [
            "NovelAI 账户信息",
            f"API 地址：{mask_url_credentials(backend.base_url)}",
            f"代理：{mask_url_credentials(proxy) if proxy else '未设置（使用 AstrBot 全局代理或直连）'}",
            f"Token：{mask_token(backend.token)}",
            f"订阅：{tier_name(tier)}（{status}）",
            "Anlas：{}（订阅 {} + 购买 {}）".format(
                total if total is not None else "未知",
                steps.get("fixedTrainingStepsLeft", "未知"),
                steps.get("purchasedTrainingSteps", "未知"),
            ),
        ]
        lines += [f"{label}：{value}" for label, value in novelai_balance_info(data).items if label == "V5 用量条"]

        orient = normalize_orientation(self.section("draw").get("default_orientation")) or "portrait"
        width, height = backend.size(orient)
        free =tier == 3 and bool(data.get("active")) and is_opus_free(backend.model, width, height, backend.steps)
        lines.append(
            f"当前参数：{backend.model} {width}x{height} {backend.steps}步 → "
            + ("符合 Opus 免费出图条件" if free else "可能消耗 Anlas")
        )
        return "\n".join(lines)
