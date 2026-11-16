"""检查版本号与更新日志，并输出 GitHub Release 的说明。

用法：
  python .github/scripts/release_notes.py          检查 metadata.yaml 的版本号在 CHANGELOG.md 中有条目（CI 使用）
  python .github/scripts/release_notes.py v1.3.3   另外检查标签与版本号一致，并输出该版本的更新内容（发布使用）
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG_RE = re.compile(r"v\d+\.\d+\.\d+")


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def metadata_version() -> str:
    match = re.search(r"^version:\s*[\"']?([^\s\"']+)[\"']?\s*$", read("metadata.yaml"), re.M)
    if not match:
        fail("metadata.yaml 中没有 version 字段")
    return match[1]


def changelog_section(version: str) -> str:
    """CHANGELOG.md 中「## v1.3.2（2026-10-30）」标题下、下一个二级标题之前的内容。"""
    pattern = rf"^## {re.escape(version)}(?![\w.-])[^\n]*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, read("CHANGELOG.md"), re.M | re.S)
    if not match:
        fail(f"CHANGELOG.md 中没有 {version} 的条目，请添加「## {version}（日期）」并写明更新内容")
    body = match[1].strip()
    if not body:
        fail(f"CHANGELOG.md 中 {version} 的条目没有内容")
    return body


def main(argv: list[str]) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows 控制台默认不是 UTF-8
        stream.reconfigure(encoding="utf-8")
    version = metadata_version()
    if not TAG_RE.fullmatch(version):
        fail(f"metadata.yaml 的版本号 {version} 格式不正确，应为 v主版本.次版本.修订号，如 v1.3.3")
    notes = changelog_section(version)
    if len(argv) < 2:
        print(f"版本号 {version} 与 CHANGELOG.md 一致")
        return
    tag = argv[1]
    if tag != version:
        fail(f"标签 {tag} 与 metadata.yaml 的版本号 {version} 不一致，请先更新 metadata.yaml 和 CHANGELOG.md")
    print(notes)
    print(f"\n完整的更新日志见 [CHANGELOG.md](https://github.com/chenxuxin188/astrbot_plugin_multi_draw/blob/{tag}/CHANGELOG.md)。")


if __name__ == "__main__":
    main(sys.argv)
