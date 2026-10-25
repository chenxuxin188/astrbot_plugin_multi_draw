"""「绘图设置」页面配置处理的单元测试：python -m pytest tests"""

import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

import settings  # noqa: E402
from backends.base import SECRET_MASK, mask_secret  # noqa: E402

SCHEMA = {
    "admin_ids": {"type": "list", "default": []},
    "draw": {
        "type": "object",
        "items": {
            "backend": {"type": "string", "default": "novelai", "options": ["novelai", "openai"]},
            "max_concurrency": {"type": "int", "default": 2},
            "save_images": {"type": "bool", "default": True},
        },
    },
    "openai": {
        "type": "object",
        "items": {
            "api_key": {"type": "string", "default": "", "secret": True},
            "model": {"type": "string", "default": "gpt-image-1"},
            "scale": {"type": "float", "default": 5.0},
            "extra_body": {"type": "text", "default": ""},
        },
    },
}


def make_config():
    return {
        "admin_ids": ["1"],
        "draw": {"backend": "novelai", "max_concurrency": 2, "save_images": True},
        "openai": {"api_key": "sk-secret-123456", "model": "gpt-image-1", "scale": 5.0, "extra_body": ""},
        "unrelated": {"keep": True},
    }


def test_mask_secret():
    assert mask_secret("") == ""
    assert mask_secret("short") == SECRET_MASK
    assert mask_secret("sk-secret-123456") == SECRET_MASK + "3456"


def test_masked_config_hides_secrets_and_fills_defaults():
    config = make_config()
    del config["draw"]["save_images"]
    out = settings.masked_config(SCHEMA, config)
    assert out["openai"]["api_key"] == SECRET_MASK + "3456"
    assert out["draw"]["save_images"] is True  # 缺失时使用默认值
    assert "unrelated" not in out
    assert config["openai"]["api_key"] == "sk-secret-123456"  # 不修改原配置


def test_merge_keeps_masked_secret_and_coerces():
    config = make_config()
    incoming = {
        "draw": {"backend": "openai", "max_concurrency": "3", "save_images": False, "bogus": 1},
        "openai": {"api_key": SECRET_MASK + "3456", "scale": "7.5", "extra_body": '{"a": 1}'},
        "admin_ids": "1\n 2 \n\n",
        "unknown_section": {"x": 1},
    }
    changed, errors = settings.merge_config(SCHEMA, config, incoming)
    assert errors == []
    assert config["draw"] == {"backend": "openai", "max_concurrency": 3, "save_images": False}
    assert config["openai"]["api_key"] == "sk-secret-123456"
    assert config["openai"]["scale"] == 7.5 and config["admin_ids"] == ["1", "2"]
    assert "unknown_section" not in config and "bogus" not in config["draw"]
    assert set(changed) == {"draw.backend", "draw.max_concurrency", "draw.save_images", "openai.scale",
                            "openai.extra_body", "admin_ids"}


def test_merge_replaces_and_clears_secret():
    config = make_config()
    settings.merge_config(SCHEMA, config, {"openai": {"api_key": " sk-new "}})
    assert config["openai"]["api_key"] == "sk-new"
    settings.merge_config(SCHEMA, config, {"openai": {"api_key": ""}})
    assert config["openai"]["api_key"] == ""


@pytest.mark.parametrize(
    "incoming",
    [
        {"draw": {"backend": "midjourney"}},
        {"draw": {"max_concurrency": "abc"}},
        {"draw": {"max_concurrency": 1.5}},
        {"draw": {"max_concurrency": True}},
        {"draw": {"save_images": "maybe"}},
        {"openai": {"scale": "nan"}},
        {"openai": {"model": {"x": 1}}},
        {"admin_ids": 5},
    ],
)
def test_merge_rejects_bad_values_atomically(incoming):
    config = make_config()
    before = json.loads(json.dumps(config))
    incoming = {**incoming, "openai": {**incoming.get("openai", {}), "model": incoming.get("openai", {}).get(
        "model", "gpt-image-1.5")}}
    changed, errors = settings.merge_config(SCHEMA, config, incoming)
    assert errors and changed == []
    assert config == before  # 有错误时不修改任何字段


def test_slider_range_clamped():
    field = {"type": "int", "default": 2, "slider": {"min": 1, "max": 8, "step": 1}}
    assert settings.coerce(field, "8", 2) == 8
    assert settings.coerce(field, 0, 2) == 1 and settings.coerce(field, "100", 2) == 8
    assert settings.coerce({"type": "float", "slider": {"min": 0, "max": 1}}, 0.5, 0) == 0.5
    assert settings.coerce({"type": "float", "slider": {"min": 0, "max": 1}}, 3, 0) == 1


def test_unchanged_values_not_revalidated():
    schema = {"draw": {"type": "object", "items": {
        "backend": {"type": "string", "default": "a", "options": ["a", "b"]},
        "max_concurrency": {"type": "int", "default": 2, "slider": {"min": 1, "max": 8}},
    }}}
    # 已保存的值超出范围 / 不在选项中（旧版本或其他页面保存的），页面只改了别的字段
    config = {"draw": {"backend": "legacy", "max_concurrency": 10}}
    changed, errors = settings.merge_config(schema, config, {"draw": {"backend": "legacy", "max_concurrency": 10}})
    assert (changed, errors) == ([], [])
    changed, errors = settings.merge_config(schema, config, {"draw": {"backend": "b", "max_concurrency": 10}})
    assert errors == [] and changed == ["draw.backend"] and config["draw"] == {"backend": "b", "max_concurrency": 10}


def test_stale_mask_keeps_current_secret():
    config = make_config()
    config["openai"]["api_key"] = "sk-NEWKEY-bbbbbbbb2222"  # 页面打开后 Key 在别处被修改
    changed, errors = settings.merge_config(SCHEMA, config, {"openai": {"api_key": SECRET_MASK + "3456",
                                                                        "model": "gpt-image-2"}})
    assert errors == [] and changed == ["openai.model"]
    assert config["openai"]["api_key"] == "sk-NEWKEY-bbbbbbbb2222"
    # 掩码后面接着粘贴了新 Key：拒绝而不是保存成错误的 Key
    changed, errors = settings.merge_config(SCHEMA, config, {"openai": {"api_key": SECRET_MASK + "3456sk-x"}})
    assert errors and changed == [] and config["openai"]["api_key"] == "sk-NEWKEY-bbbbbbbb2222"
    draft = settings.draft_section(SCHEMA, config, "openai", {"api_key": SECRET_MASK + "1111"})
    assert draft["api_key"] == "sk-NEWKEY-bbbbbbbb2222"


def test_merge_rejects_non_dict():
    assert settings.merge_config(SCHEMA, make_config(), None) == ([], ["配置格式错误"])


def test_draft_section_overlays_unsaved_values():
    config = make_config()
    draft = {"api_key": SECRET_MASK + "3456", "model": "gpt-image-1.5", "scale": "oops"}
    merged = settings.draft_section(SCHEMA, config, "openai", draft)
    assert merged["api_key"] == "sk-secret-123456"  # 掩码还原为已保存的 Key
    assert merged["model"] == "gpt-image-1.5"
    assert merged["scale"] == 5.0  # 类型不对时沿用已保存的值
    assert config["openai"]["model"] == "gpt-image-1"  # 不修改已保存的配置
    fresh = settings.draft_section(SCHEMA, {}, "openai", {"api_key": "sk-typed"})
    assert fresh["api_key"] == "sk-typed" and fresh["model"] == "gpt-image-1"


def test_real_schema_is_consistent():
    schema = settings.load_schema(PLUGIN_DIR)
    for section, key, field in settings._fields(schema):
        assert field.get("type") in ("string", "text", "int", "float", "bool", "list"), (section, key)
        default = field.get("default")
        if "options" in field:
            assert default in field["options"], (section, key)
            if "labels" in field:
                assert len(field["labels"]) == len(field["options"]), (section, key)
        # 默认值必须能通过页面保存时的类型校验
        assert settings.coerce(field, default, default) == default, (section, key)
    assert schema["draw"]["items"]["backend"]["options"][0] == "novelai"


def test_non_finite_int_rejected_cleanly():
    field = {"type": "int", "default": 2}
    for bad in ("1e400", "9" * 400, "nan", float("inf")):
        with pytest.raises(ValueError, match="应为整数"):
            settings.coerce(field, bad, 2)


def test_draft_section_rejects_mask_with_appended_text():
    config = make_config()
    with pytest.raises(ValueError, match="掩码"):
        settings.draft_section(SCHEMA, config, "openai", {"api_key": SECRET_MASK + "3456sk-typed"})
