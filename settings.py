"""「绘图设置」页面使用的配置处理：按 _conf_schema.json 校验类型、隐藏与还原敏感字段。

本模块不依赖 AstrBot，便于单独测试。
"""

import copy
import json
import math
from pathlib import Path

try:
    from .backends.base import SECRET_MASK, mask_secret
except ImportError:  # 单独测试时作为顶层模块导入
    from backends.base import SECRET_MASK, mask_secret


def load_schema(plugin_dir: Path) -> dict:
    return json.loads((Path(plugin_dir) / "_conf_schema.json").read_text(encoding="utf-8"))


def _fields(schema: dict):
    """遍历 (section, key, field)；顶层非 object 字段的 section 为 None。"""
    for name, field in schema.items():
        if field.get("type") == "object":
            for key, sub in (field.get("items") or {}).items():
                yield name, key, sub
        else:
            yield None, name, field


def _get(config: dict, section: str | None, key: str):
    if section is None:
        return config.get(key)
    value = config.get(section)
    return value.get(key) if isinstance(value, dict) else None


def masked_config(schema: dict, config: dict) -> dict:
    """返回只包含 schema 中字段的配置副本，敏感字段替换为掩码。"""
    out: dict = {}
    for section, key, field in _fields(schema):
        value = _get(config, section, key)
        if value is None:
            value = copy.deepcopy(field.get("default"))
        if field.get("secret") and isinstance(value, str):
            value = mask_secret(value)
        target = out if section is None else out.setdefault(section, {})
        target[key] = copy.deepcopy(value)
    return out


def _clamp(field: dict, number: float) -> float:
    """有 slider 的数值字段限制在 slider 的范围内（与 AstrBot 自带配置页面和运行时的处理一致）。"""
    slider = field.get("slider") if isinstance(field.get("slider"), dict) else {}
    lo, hi = slider.get("min"), slider.get("max")
    if lo is not None and number < lo:
        number = lo
    if hi is not None and number > hi:
        number = hi
    return number


def coerce(field: dict, value, current):
    """把页面提交的值转换成字段声明的类型；无法转换时抛出 ValueError（说明给用户看）。"""
    kind = field.get("type", "string")
    if field.get("secret") and isinstance(value, str) and SECRET_MASK[0] in value:
        stripped = value.strip()
        if stripped.startswith(SECRET_MASK) and len(stripped) <= len(SECRET_MASK) + 4:
            # 页面上没有修改的敏感字段（掩码）：保留原值。即使页面打开后 Key 在别处被修改过、
            # 掩码已经对不上，也不能把掩码当成新的 Key 保存
            return current if isinstance(current, str) else ""
        raise ValueError("中含有掩码字符（•），请清空后重新粘贴完整的值")
    if kind in ("string", "text"):
        if value is None:
            value = ""
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError("应为文本")
        value = str(value)
        if kind == "string":
            value = value.strip()
        options = field.get("options")
        if options and value not in options:
            raise ValueError(f"只能是 {', '.join(map(str, options))} 之一")
        return value
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError("应为开关（true / false）")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError("应为整数")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("应为整数") from None
        if not math.isfinite(number) or number != int(number):
            raise ValueError("应为整数")
        return int(_clamp(field, number))
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError("应为数字")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("应为数字") from None
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError("应为数字")
        return _clamp(field, number)
    if kind == "list":
        if isinstance(value, str):
            value = [line for line in value.splitlines()]
        if not isinstance(value, list):
            raise ValueError("应为列表")
        return [str(v).strip() for v in value if str(v).strip()]
    raise ValueError(f"不支持的配置类型 {kind}")


def merge_config(schema: dict, config: dict, incoming: dict) -> tuple[list[str], list[str]]:
    """把页面提交的配置合并进 config（原地修改），只接受 schema 中声明的字段。

    返回 (修改过的字段列表, 错误列表)。有错误时不修改 config。
    """
    if not isinstance(incoming, dict):
        return [], ["配置格式错误"]
    updates: list[tuple[str | None, str, object]] = []
    errors: list[str] = []
    for section, key, field in _fields(schema):
        if section is None:
            if key not in incoming:
                continue
            value = incoming[key]
        else:
            part = incoming.get(section)
            if not isinstance(part, dict) or key not in part:
                continue
            value = part[key]
        current = _get(config, section, key)
        if value == current:
            continue  # 没有修改的字段不重新校验（可能是旧版本或其他页面保存的值）
        label = f"{section}.{key}" if section else key
        try:
            new = coerce(field, value, current)
        except ValueError as e:
            errors.append(f"{field.get('description') or label}（{label}）{e}")
            continue
        if new != current:
            updates.append((section, key, new))
    if errors:
        return [], errors
    changed = []
    for section, key, new in updates:
        if section is None:
            config[key] = new
        else:
            part = config.get(section)
            if not isinstance(part, dict):
                part = {}
                config[section] = part
            part[key] = new
        changed.append(f"{section}.{key}" if section else key)
    return changed, []


def draft_section(schema: dict, config: dict, name: str, draft) -> dict:
    """试画 / 获取模型列表时使用的平台配置：已保存的配置叠加页面上尚未保存的修改。

    未修改的敏感字段（值等于掩码）还原为已保存的值；类型不对的字段沿用已保存的值。
    敏感字段填写有误（如掩码后面接着粘贴了新 Key）时抛出 ValueError，不能悄悄改用已保存的值。
    """
    saved = config.get(name) if isinstance(config.get(name), dict) else {}
    merged = copy.deepcopy(saved)
    items = (schema.get(name) or {}).get("items") or {}
    for key, field in items.items():
        if key not in merged:
            merged[key] = copy.deepcopy(field.get("default"))
    if isinstance(draft, dict):
        for key, field in items.items():
            if key not in draft:
                continue
            try:
                merged[key] = coerce(field, draft[key], saved.get(key))
            except ValueError as e:
                if field.get("secret"):
                    raise ValueError(f"{field.get('description') or key}{e}") from None
                continue
    return merged
