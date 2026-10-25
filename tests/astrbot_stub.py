"""测试用的最小 astrbot 模拟包：不安装 AstrBot 也能导入并运行 main.py / webapi.py。

导入插件包之前调用 install(data_dir)。
"""

import json
import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("astrbot-stub")


class AstrBotConfig(dict):
    def __init__(self, data=None, path=None):
        super().__init__(data or {})
        object.__setattr__(self, "config_path", path)
        object.__setattr__(self, "saves", 0)

    def save_config(self, replace_config=None):
        if replace_config:
            self.update(replace_config)
        object.__setattr__(self, "saves", self.saves + 1)
        if self.config_path:
            Path(self.config_path).write_text(json.dumps(self, ensure_ascii=False, indent=2), encoding="utf-8")


class MessageEventResult(list):
    def use_t2i(self, flag):
        return self


class Image:
    def __init__(self, file):
        self.file = file

    @classmethod
    def fromBytes(cls, data: bytes):
        obj = cls("base64://...")
        obj.data = data
        return obj


class Plain:
    def __init__(self, text):
        self.text = text


class AstrMessageEvent:
    def __init__(self, sender="10001", group="20002", platform="aiocqhttp"):
        self.sender = sender
        self.group = group
        self.platform = platform
        self.unified_msg_origin = f"{platform}:GroupMessage:{group}"
        self.sent: list = []
        self._extra = {}

    def get_sender_id(self):
        return self.sender

    def get_sender_name(self):
        return "tester"

    def get_group_id(self):
        return self.group

    def get_platform_name(self):
        return self.platform

    def get_extra(self, key=None, default=None):
        return self._extra.get(key, default)

    def set_extra(self, key, value):
        self._extra[key] = value

    def plain_result(self, text):
        return MessageEventResult([Plain(text)])

    def chain_result(self, chain):
        return MessageEventResult(chain)

    async def send(self, result):
        if getattr(self, "closed", False):
            return  # 模拟 WebChat 本轮结束后回复通道已关闭：消息被丢弃
        self.sent.append(result)


class _Group:
    def __init__(self, fn):
        self.fn = fn

    def command(self, name):
        return lambda fn: fn


class filter:  # noqa: N801
    @staticmethod
    def llm_tool(name=None):
        return lambda fn: fn

    @staticmethod
    def command_group(name):
        return lambda fn: _Group(fn)


class Star:
    def __init__(self, context):
        self.context = context


class StarTools:
    data_dir: Path = Path(".")

    @classmethod
    def get_data_dir(cls, name=None):
        path = cls.data_dir / (name or "plugin")
        path.mkdir(parents=True, exist_ok=True)
        return path


class LLMResponse:
    def __init__(self, text, role="assistant"):
        self.completion_text = text
        self.role = role


class Context:
    """Fake Context: llm_generate is answered by `responder(kwargs) -> str` (raise to simulate failure)."""

    def __init__(self, responder, providers=None):
        self.responder = responder
        self.calls: list[dict] = []
        self.proactive: list = []  # context.send_message 主动发送的消息
        self.config: dict = {}
        self.registered_web_apis: list = []
        self.persona_manager = None
        self.providers = providers or {}

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(self.responder(kwargs))

    async def get_current_chat_provider_id(self, umo):
        return "session-llm"

    def get_provider_by_id(self, pid):
        return self.providers.get(pid)

    def get_all_providers(self):
        return list(self.providers.values())

    def register_web_api(self, route, handler, methods, desc):
        for i, api in enumerate(self.registered_web_apis):
            if api[0] == route and api[2] == methods:
                self.registered_web_apis[i] = (route, handler, methods, desc)
                return
        self.registered_web_apis.append((route, handler, methods, desc))

    def get_config(self, umo=None):
        return self.config

    async def send_message(self, session, message_chain):
        self.proactive.append((session, message_chain))
        return True


class FakeRequest:
    def __init__(self):
        self.body = {}
        self.query = {}

    async def json(self, default=None):
        return self.body if self.body is not None else default


request = FakeRequest()


def install(data_dir: Path):
    StarTools.data_dir = Path(data_dir)
    root = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = AstrBotConfig
    api.logger = logger
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = AstrMessageEvent
    event.filter = filter
    comps = types.ModuleType("astrbot.api.message_components")
    comps.Image = Image
    comps.Plain = Plain
    star = types.ModuleType("astrbot.api.star")
    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    web = types.ModuleType("astrbot.api.web")
    web.request = request
    root.api = api
    for name, mod in {"astrbot": root, "astrbot.api": api, "astrbot.api.event": event,
                      "astrbot.api.message_components": comps, "astrbot.api.star": star,
                      "astrbot.api.web": web}.items():
        sys.modules[name] = mod
