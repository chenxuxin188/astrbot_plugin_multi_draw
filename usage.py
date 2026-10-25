"""绘图消耗：按平台配置给每条绘图记录算出费用，并按平台汇总一段时间内的绘图记录。"""

import math
from datetime import date, datetime, time, timedelta

try:
    from .backends.base import as_bool, as_float, as_str
except ImportError:  # 单独测试时作为顶层模块导入
    from backends.base import as_bool, as_float, as_str

TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def estimate_cost(usage: dict, section: dict) -> tuple[float | None, str, bool]:
    """返回 (费用, 货币单位, 是否为估算)。

    平台直接返回了费用（xAI）时使用它；否则按配置的 token 价格（需要响应中有输入 / 输出 token 数），
    再否则按每张图片的价格估算。都没有配置时返回 (None, "", False)。
    """
    cost = _number(usage.get("cost"))
    if cost is not None and as_str(usage.get("currency")):
        return cost, as_str(usage["currency"]), False
    currency = as_str(section.get("price_currency")) or "USD"
    price_in = as_float(section.get("price_input_per_million"), 0.0, 0.0)
    price_out = as_float(section.get("price_output_per_million"), 0.0, 0.0)
    if price_in and price_out and ("input_tokens" in usage or "output_tokens" in usage):
        cost = (_int(usage.get("input_tokens")) * price_in + _int(usage.get("output_tokens")) * price_out) / 1e6
        return round(cost, 6), currency, True
    per_image = as_float(section.get("price_per_image"), 0.0, 0.0)
    if per_image:
        return round(per_image * max(1, _int(usage.get("images"))), 6), currency, True
    return None, "", False


def apply_usage(record: dict, section: dict) -> None:
    """按平台的「记录用量」设置处理记录中的 usage：关闭时去掉，开启时写入费用（cost / currency / cost_estimated）。"""
    usage = record.get("usage")
    if not isinstance(usage, dict):
        record.pop("usage", None)
        return
    if not as_bool(section.get("track_usage"), True):
        record.pop("usage", None)
        return
    cost, currency, estimated = estimate_cost(usage, section)
    if cost is not None:
        record.update(cost=cost, currency=currency)
        if estimated:
            record["cost_estimated"] = True


def _number(value) -> float | None:
    """记录中的数字：不是有限的数（如手动编辑出的 NaN、Infinity）时返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _int(value) -> int:
    number = _number(value)
    return max(0, int(number)) if number is not None else 0


def format_cost(cost: float, currency: str, estimated: bool = False) -> str:
    text = f"{cost:.4f}".rstrip("0").rstrip(".") if cost else "0"
    if text == "0" and cost:
        text = "<0.0001"
    return f"{'约 ' if estimated else ''}{text} {currency}".strip()


def format_record_usage(record: dict) -> str:
    """一条记录的用量和费用，如「1 张 · 4,160 tokens · 约 0.17 USD」。没有用量时返回空字符串。"""
    usage = record.get("usage")
    parts = []
    if isinstance(usage, dict):
        if _int(usage.get("images")) > 1:
            parts.append(f"{_int(usage['images'])} 张")
        if _int(usage.get("total_tokens")):
            parts.append(f"{_int(usage['total_tokens']):,} tokens")
    cost = _number(record.get("cost"))
    if cost is not None and record.get("currency"):
        parts.append(format_cost(cost, as_str(record["currency"]), bool(record.get("cost_estimated"))))
    return " · ".join(parts)


# ---------------------------------------------------------------- 汇总


def period_start(days: int, now: datetime | None = None) -> datetime | None:
    """统计的起始时间：1 表示今天 0 点起，7 表示含今天在内的最近 7 天，0 表示不限。"""
    if days <= 0:
        return None
    if now is None:
        # 按起始那天的本地时区偏移换算，跨夏令时切换时也是当天 0 点
        return datetime.combine(date.today() - timedelta(days=days - 1), time.min).astimezone()
    return datetime.combine(now.date() - timedelta(days=days - 1), time.min, tzinfo=now.tzinfo)


def period_label(days: int) -> str:
    if days <= 0:
        return "全部记录"
    return "今天" if days == 1 else f"最近 {days} 天"


def record_time(record: dict) -> datetime | None:
    try:
        when = datetime.fromisoformat(as_str(record.get("time")))
        return when if when.tzinfo else when.astimezone()
    except (ValueError, OverflowError, OSError):  # 时间格式错误，或没有时区且超出本机能换算的范围
        return None


def _new_stats(key: str, label: str) -> dict:
    return {"backend": key, "label": label, "success": 0, "failed": 0, "images": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "anlas": 0, "anlas_records": 0, "costs": {}, "estimated": []}


def _add(stats: dict, record: dict) -> None:
    if record.get("status") != "success":
        stats["failed"] += 1
    else:
        stats["success"] += 1
        usage = record.get("usage")
        images = _int(usage.get("images")) if isinstance(usage, dict) else 0
        stats["images"] += images or 1
    usage = record.get("usage")
    if isinstance(usage, dict):
        for key in TOKEN_KEYS:
            stats[key] += _int(usage.get(key))
    anlas = _number(record.get("anlas_cost"))
    if anlas is not None:
        stats["anlas_records"] += 1
        stats["anlas"] += max(0, int(anlas))  # 生成期间余额增加（如充值、月初刷新）时不算消耗
    cost = _number(record.get("cost"))
    currency = as_str(record.get("currency"))
    if cost is not None and currency:
        stats["costs"][currency] = stats["costs"].get(currency, 0.0) + cost
        if record.get("cost_estimated") and currency not in stats["estimated"]:
            stats["estimated"].append(currency)


def summarize(records: list[dict], since: datetime | None = None, user_id: str | None = None,
              backend: str | None = None, order: list[str] | tuple = (),
              trimmed_until: datetime | None = None) -> dict:
    """按平台汇总绘图记录。records 为从旧到新的全部记录，trimmed_until 为已删除的旧记录中最新一条的时间。

    返回 {"backends": [每个平台的统计], "total": 合计, "count": 统计的记录数, "truncated": 是否可能缺少更早的记录}。
    各平台的统计包括成功 / 失败次数、生成的图片数、token 数、Anlas 消耗和按货币单位分开的费用。
    """
    per: dict[str, dict] = {}
    total = _new_stats("", "合计")
    count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if user_id is not None and str(record.get("user_id", "")) != user_id:
            continue
        key = as_str(record.get("backend")) or "unknown"
        if backend and key != backend:
            continue
        if since is not None:
            when = record_time(record)
            if when is None or when < since:
                continue
        stats = per.get(key)
        if stats is None:
            stats = per[key] = _new_stats(key, as_str(record.get("backend_label")) or key)
        _add(stats, record)
        _add(total, record)
        count += 1
    rank = {k: i for i, k in enumerate(order)}
    rows = sorted(per.values(), key=lambda s: (rank.get(s["backend"], len(rank)), s["backend"]))
    for stats in [*rows, total]:
        stats["costs"] = {k: round(v, 6) for k, v in stats["costs"].items()}
    # 记录超过条数上限后会删除最早的记录：删除的记录在统计范围内时，统计不完整
    truncated = trimmed_until is not None and (since is None or trimmed_until >= since)
    return {"backends": rows, "total": total, "count": count, "truncated": truncated}


def _stats_text(stats: dict) -> str:
    parts = [f"成功 {stats['images']} 张" if stats["success"] else "没有成功的图片"]
    if stats["failed"]:
        parts[0] += f"，失败 {stats['failed']} 次"
    if stats["anlas_records"]:
        parts.append(f"消耗 {stats['anlas']:,} Anlas")
    if stats["total_tokens"]:
        tokens = f"{stats['total_tokens']:,} tokens"
        if stats["input_tokens"] or stats["output_tokens"]:
            tokens += f"（输入 {stats['input_tokens']:,} / 输出 {stats['output_tokens']:,}）"
        parts.append(tokens)
    if stats["costs"]:
        parts.append(" + ".join(format_cost(v, k, k in stats["estimated"]) for k, v in stats["costs"].items()))
    return "；".join(parts)


def format_summary(summary: dict, days: int, whose: str = "") -> str:
    """把 summarize() 的结果格式化成多行文字。whose 如「你的」，为空表示所有人。"""
    title = f"{whose}绘图消耗（{period_label(days)}）"
    if not summary["count"]:
        return f"{title}：没有绘图记录。"
    lines = [f"{title}，共 {summary['count']} 次绘图："]
    for stats in summary["backends"]:
        lines.append(f"【{stats['label']}】{_stats_text(stats)}")
    if len(summary["backends"]) > 1:
        lines.append(f"合计：{_stats_text(summary['total'])}")
    if summary["total"]["estimated"]:
        lines.append("「约」表示按配置的价格估算，实际以平台账单为准。")
    if summary["truncated"]:
        lines.append("绘图记录有条数上限，更早的记录已被删除，没有计入。")
    return "\n".join(lines)
