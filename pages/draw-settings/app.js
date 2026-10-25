/* 多平台绘图 · 绘图设置 —— AstrBot 插件页面脚本。
 *
 * 原生 ES module，无构建步骤、无外部依赖（服务器可能离线或在国内）。
 * 所有代码放在这一个文件里：旧版 AstrBot 用 query token 给页面资源鉴权，模块内的相对 import 不会带上 token。
 * 页面运行在受限 iframe 中（无 same-origin、无 allow-popups / allow-modals）：不使用 localStorage、alert、confirm，
 * 外链打不开时改为复制链接。服务器返回的文字一律通过 textContent / 属性写入，不拼接 HTML。
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const AUDIT_PAGE_SIZE = 20;
const GALLERY_LIMIT = 60;
const IMAGE_CONCURRENCY = 4;
const THUMB_EDGE = 512;
const FULL_CACHE_SIZE = 6;
const TEXTAREA_MAX = 420;

const TABS = [
  { key: "backends", label: "绘图平台" },
  { key: "prompts", label: "提示词与人格" },
  { key: "records", label: "记录" },
  { key: "gallery", label: "图库" },
];
const PROMPT_SECTIONS = ["draw", "prompt", "persona"];
const RECORD_SECTIONS = ["audit"];
const ORIENTATIONS = [
  ["portrait", "竖图"],
  ["landscape", "横图"],
  ["square", "方图"],
];
const STAGE_NAMES = { config: "检查配置", prompt: "生成提示词", generate: "生成图片", send: "发送图片" };
const SOURCE_NAMES = { llm: "LLM 调用", command: "指令" };
const PROMPT_MODES = { tags: "标签提示词", natural: "自然语言提示词" };
const PARAM_LABELS = {
  model: "模型",
  size: "尺寸",
  steps: "步数",
  scale: "CFG",
  sampler: "采样器",
  seed: "种子",
  opus_free: "Opus 免费",
  quality: "质量",
  aspect_ratio: "比例",
  resolution: "分辨率",
};
const FOLLOW_SESSION = "（留空：跟随当前会话）";
const EMPTY_LABELS = { "persona.vision_provider_id": "（留空：同「生成人格台词使用的 LLM」）" };
// 需要突出显示的开关，以及只在它开启时才生效的字段
const FEATURED = {
  "persona.comment_with_image": {
    icon: "image",
    dependents: ["vision_provider_id"],
    disabledNote: "开启「评价时附上生成的图片」后才会使用此项。",
  },
};
const SAMPLE_PROMPTS = {
  tags:
    "1girl, solo, long silver hair, blue eyes, school uniform, cherry blossoms, falling petals, smile, " +
    "looking at viewer, upper body, soft lighting",
  natural:
    "A cheerful anime girl with long silver hair and blue eyes, wearing a school uniform, standing under " +
    "blooming cherry trees as petals drift in the breeze. Soft morning light, pastel colors, detailed illustration.",
};
const SAMPLE_NEGATIVE = "lowres, bad anatomy, bad hands, text, watermark";
const TEST_NOTE = "试画使用当前页面上的设置（包括尚未保存的修改），直接把提示词发给绘图平台，不经过 LLM；会正常计费。";
const MIME_EXT = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp" };

const ICONS = {
  eye: ["M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z", "M15 12a3 3 0 1 1-6 0 3 3 0 1 1 6 0z"],
  eyeOff: [
    "M3 3l18 18",
    "M10.6 5.1A10.4 10.4 0 0 1 12 5c6.5 0 10 7 10 7a17.6 17.6 0 0 1-3.2 4.2",
    "M6.6 6.6C3.9 8.4 2 12 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6",
    "M9.9 9.9a3 3 0 0 0 4.2 4.2",
  ],
  external: ["M14 4h6v6", "M20 4l-9 9", "M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"],
  refresh: ["M20 11a8 8 0 0 0-14.9-4", "M4 3v4h4", "M4 13a8 8 0 0 0 14.9 4", "M20 21v-4h-4"],
  x: ["M6 6l12 12", "M18 6L6 18"],
  left: ["M15 6l-6 6 6 6"],
  right: ["M9 6l6 6-6 6"],
  download: ["M12 4v11", "M7 10l5 5 5-5", "M5 20h14"],
  list: ["M9 6h11", "M9 12h11", "M9 18h11", "M4.5 6h.01", "M4.5 12h.01", "M4.5 18h.01"],
  image: [
    "M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z",
    "M9.5 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z",
    "M20 15l-5-5L5 20",
  ],
  play: ["M7 4.5v15l12-7.5z"],
  copy: ["M10 9h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1z", "M5 15V5a1 1 0 0 1 1-1h10"],
  brush: [
    "M9.06 11.9l8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08",
    "M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z",
  ],
};

const S = {
  bridge: null,
  version: "",
  schema: {},
  saved: {}, // 服务器上已保存的配置（敏感字段为掩码）
  draft: {}, // 页面上正在编辑的配置
  backends: [],
  providers: [],
  personas: [],
  secretMask: "••••",
  tab: "backends",
  selected: "",
  models: {}, // 平台 key -> { loading, list, error }
  tests: {}, // 平台 key -> 试画状态
  audit: { page: 1, data: null, loading: false, error: "", seq: 0, settingsOpen: false },
  gallery: {
    list: null,
    loading: false,
    error: "",
    thumbs: new Map(),
    failed: new Map(),
    full: new Map(),
    inflight: new Map(),
    queue: [],
    pending: new Set(),
    active: 0,
  },
  saving: false,
  saveError: "",
  watchers: [],
  chrome: {},
  ui: {},
  footer: null,
  viewer: null,
};

// ------------------------------------------------------------------ 通用工具

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    Object.keys(props).forEach((k) => {
      const v = props[k];
      if (v === undefined || v === null || v === false) return;
      if (k === "class") el.className = v;
      else if (k.slice(0, 2) === "on" && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "value" || k === "checked" || k === "disabled" || k === "hidden" || k === "readOnly") el[k] = v;
      else el.setAttribute(k, v === true ? "" : String(v));
    });
  }
  children.forEach((c) => append(el, c));
  return el;
}

function append(el, child) {
  if (child === null || child === undefined || child === false) return;
  if (Array.isArray(child)) {
    child.forEach((c) => append(el, c));
    return;
  }
  el.appendChild(typeof child === "object" ? child : document.createTextNode(String(child)));
}

function setChildren(el, ...children) {
  el.textContent = "";
  children.forEach((c) => append(el, c));
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  (ICONS[name] || []).forEach((d) => {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.appendChild(path);
  });
  return svg;
}

function spinner(extra) {
  return h("span", { class: "spinner" + (extra ? " " + extra : ""), "aria-hidden": "true" });
}

let uidSeq = 0;
function uid(prefix) {
  uidSeq += 1;
  return String(prefix).replace(/[^A-Za-z0-9_-]/g, "_") + "-" + uidSeq;
}

function isObj(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function clone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
}

function strOf(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "string" ? v : String(v);
}

function asBool(v) {
  return v === true || v === "true";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function debounce(fn, ms) {
  let timer = 0;
  return function () {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

function withTimeout(promise, ms, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    Promise.resolve(promise).then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

function fmtTime(iso) {
  return strOf(iso).slice(0, 19).replace("T", " ");
}

function fmtSec(n) {
  const v = Number(n);
  return isFinite(v) ? (Math.round(v * 10) / 10).toFixed(1) : "?";
}

function fmtBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return v + " B";
  if (v < 1024 * 1024) return (v / 1024).toFixed(1) + " KB";
  return (v / 1024 / 1024).toFixed(2) + " MB";
}

function pad2(n) {
  return (n < 10 ? "0" : "") + n;
}

function stamp() {
  const d = new Date();
  return (
    d.getFullYear() + pad2(d.getMonth() + 1) + pad2(d.getDate()) + "_" + pad2(d.getHours()) + pad2(d.getMinutes()) + pad2(d.getSeconds())
  );
}

function isImageDataUrl(s) {
  return typeof s === "string" && /^data:image\/[a-z0-9.+-]+;base64,/i.test(s);
}

function mimeOf(dataUrl) {
  const m = /^data:([^;,]+)/.exec(strOf(dataUrl));
  return m ? m[1].toLowerCase() : "";
}

function extOf(mime) {
  return MIME_EXT[strOf(mime).toLowerCase()] || "png";
}

function dataUrlBytes(s) {
  const i = strOf(s).indexOf(",");
  if (i < 0) return 0;
  const len = s.length - i - 1;
  const padding = s.slice(-2) === "==" ? 2 : s.slice(-1) === "=" ? 1 : 0;
  return Math.max(0, Math.floor((len * 3) / 4) - padding);
}

function fmtParam(v) {
  if (v === true) return "是";
  if (v === false) return "否";
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function errorText(err) {
  const msg = strOf(err && err.message !== undefined ? err.message : err).trim();
  if (!msg) return "请求失败";
  if (/status code 404/i.test(msg)) return "接口不存在（404）：请确认插件已启用、版本与页面一致，然后刷新页面。";
  if (/network error/i.test(msg)) return "无法连接到 AstrBot，请检查网络后重试。";
  return msg;
}

function copyText(text) {
  const fallback = () => {
    try {
      const ta = h("textarea", { readOnly: true, class: "sr-only" });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch (e) {
      return false;
    }
  };
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(() => true, fallback);
    }
  } catch (e) {
    // 继续使用 execCommand
  }
  return Promise.resolve(fallback());
}

function autoGrow(ta) {
  if (!ta.isConnected || ta.offsetParent === null) return;
  ta.style.height = "auto";
  const full = ta.scrollHeight + 2;
  ta.style.height = Math.min(full, TEXTAREA_MAX) + "px";
  ta.classList.toggle("is-capped", full > TEXTAREA_MAX);
}

function growTextareas(root) {
  Array.prototype.forEach.call(root.querySelectorAll("textarea"), autoGrow);
}

// ------------------------------------------------------------------ bridge 与接口

function api(method, endpoint, payload) {
  let call;
  try {
    call = method === "get" ? S.bridge.apiGet(endpoint, payload || {}) : S.bridge.apiPost(endpoint, payload || {});
  } catch (e) {
    return Promise.reject(new Error(errorText(e)));
  }
  return Promise.resolve(call).then(
    (res) => {
      if (!isObj(res)) throw new Error("服务器返回了无法识别的数据");
      return res;
    },
    (e) => {
      throw new Error(errorText(e));
    },
  );
}

async function waitForBridge() {
  for (let i = 0; i < 40; i++) {
    const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;
    if (bridge && typeof bridge.apiGet === "function") return bridge;
    await sleep(100);
  }
  return null;
}

function applyContextTheme(ctx) {
  if (ctx && typeof ctx.isDark === "boolean") {
    document.documentElement.setAttribute("data-theme", ctx.isDark ? "dark" : "light");
  }
}

// ------------------------------------------------------------------ 配置读写

function isSection(field) {
  return isObj(field) && field.type === "object";
}

function sectionItems(name) {
  const f = S.schema[name];
  return isSection(f) && isObj(f.items) ? f.items : {};
}

function readCfg(cfg, sec, key) {
  if (!sec) return cfg[key];
  return isObj(cfg[sec]) ? cfg[sec][key] : undefined;
}

function getDraft(sec, key) {
  return readCfg(S.draft, sec, key);
}

function setDraft(sec, key, value) {
  if (sec) {
    if (!isObj(S.draft[sec])) S.draft[sec] = {};
    S.draft[sec][key] = value;
  } else {
    S.draft[key] = value;
  }
  S.saveError = "";
  notify();
}

function draftBackend() {
  return strOf(getDraft("draw", "backend"));
}

function savedBackend() {
  return strOf(readCfg(S.saved, "draw", "backend"));
}

function findBackend(key) {
  for (let i = 0; i < S.backends.length; i++) if (S.backends[i].key === key) return S.backends[i];
  return null;
}

function backendLabel(key) {
  if (!key) return "未设置";
  const b = findBackend(key);
  if (b) return strOf(b.label) || key;
  const f = sectionItems("draw").backend;
  if (f && Array.isArray(f.options)) {
    const i = f.options.indexOf(key);
    if (i >= 0 && Array.isArray(f.labels) && f.labels[i]) return strOf(f.labels[i]);
  }
  return key;
}

function changedSections() {
  return Object.keys(S.schema).filter((k) => !deepEqual(S.draft[k], S.saved[k]));
}

function sectionTitle(name) {
  const f = S.schema[name];
  return (isObj(f) && strOf(f.description)) || name;
}

function sectionTab(name) {
  if (findBackend(name)) return "backends";
  if (RECORD_SECTIONS.indexOf(name) >= 0 && isSection(S.schema[name])) return "records";
  return "prompts";
}

/** 阻止保存的错误（只检查数字类字段）。 */
function fieldError(field, value) {
  if (field.type === "int") return typeof value === "number" && Number.isInteger(value) ? "" : "请填写整数";
  if (field.type === "float") return typeof value === "number" && isFinite(value) ? "" : "请填写数字";
  return "";
}

/** 保存前检查改动过的字段。 */
function draftErrors(sections) {
  const out = [];
  const check = (sec, key, field, prefix) => {
    const value = getDraft(sec, key);
    if (deepEqual(value, readCfg(S.saved, sec, key))) return;
    const err = fieldError(field, value);
    if (err) out.push(prefix + (field.description || key) + "：" + err);
  };
  sections.forEach((name) => {
    const f = S.schema[name];
    if (isSection(f)) {
      const items = sectionItems(name);
      Object.keys(items).forEach((key) => check(name, key, items[key], sectionTitle(name) + " · "));
    } else if (isObj(f)) {
      check(null, name, f, "");
    }
  });
  return out;
}

// 依赖草稿的界面片段在这里登记；元素离开页面后自动丢弃
function watch(owner, fn) {
  S.watchers.push({ owner, fn });
  fn();
}

function notify() {
  S.watchers = S.watchers.filter((w) => w.owner.isConnected);
  S.watchers.slice().forEach((w) => {
    try {
      w.fn();
    } catch (e) {
      console.error("[multi_draw]", e);
    }
  });
  refreshChrome();
}

// ------------------------------------------------------------------ 提示与浮层

function toast(text, type) {
  const box = document.getElementById("toasts");
  if (!box) return;
  const el = h("div", { class: "toast" + (type ? " toast-" + type : "") }, text);
  box.appendChild(el);
  while (box.children.length > 3) box.removeChild(box.firstChild);
  setTimeout(() => el.remove(), type === "error" ? 6500 : 3000);
}

function closeViewer() {
  if (S.viewer) S.viewer.close();
}

/** 查看大图。items: [{ src? | load(), name?, title?, time?, sizeBytes?, downloadName? }] */
function openViewer(items, index) {
  closeViewer();
  if (!items.length) return;
  const prevFocus = document.activeElement;
  const multi = items.length > 1;
  const stage = h("div", { class: "overlay-box" });
  const closeBtn = h("button", { type: "button", class: "overlay-close", "aria-label": "关闭（Esc）", title: "关闭" }, icon("x"));
  const prevBtn = multi
    ? h("button", { type: "button", class: "overlay-nav overlay-prev", "aria-label": "上一张", title: "上一张（←）" }, icon("left"))
    : null;
  const nextBtn = multi
    ? h("button", { type: "button", class: "overlay-nav overlay-next", "aria-label": "下一张", title: "下一张（→）" }, icon("right"))
    : null;
  const root = h("div", { class: "overlay", role: "dialog", "aria-modal": "true", "aria-label": "查看图片" }, stage, closeBtn, prevBtn, nextBtn);
  let cur = index;
  let seq = 0;

  function fillInfo(info, it, src, img) {
    const parts = [];
    if (it.title) parts.push(h("span", null, it.title));
    if (it.name) parts.push(h("span", { class: "mono" }, it.name));
    if (it.time) parts.push(h("span", null, fmtTime(it.time)));
    if (img && img.naturalWidth) parts.push(h("span", null, img.naturalWidth + " × " + img.naturalHeight));
    const bytes = Number(it.sizeBytes) || (src ? dataUrlBytes(src) : 0);
    if (bytes) parts.push(h("span", null, fmtBytes(bytes)));
    if (multi) parts.push(h("span", null, cur + 1 + " / " + items.length));
    if (src) {
      const fname = it.downloadName || (it.name ? strOf(it.name).replace(/[\\/]/g, "_") : "image_" + stamp() + "." + extOf(mimeOf(src)));
      parts.push(h("a", { href: src, download: fname }, icon("download"), "下载"));
    }
    setChildren(info, parts);
  }

  function show(i) {
    cur = (i + items.length) % items.length;
    const it = items[cur];
    const my = ++seq;
    const info = h("div", { class: "overlay-info" });
    setChildren(stage, h("div", { class: "overlay-loading" }, spinner(), "正在加载图片…"), info);
    fillInfo(info, it, null, null);
    const loader = it.src ? Promise.resolve(it.src) : it.load();
    loader.then(
      (src) => {
        if (my !== seq || !root.isConnected) return;
        if (!isImageDataUrl(src)) throw new Error("图片格式无效");
        const img = h("img", { class: "overlay-img", alt: it.title || it.name || "图片" });
        img.addEventListener("load", () => {
          if (my === seq) fillInfo(info, it, src, img);
        });
        img.src = src;
        setChildren(stage, img, info);
        fillInfo(info, it, src, null);
      },
    ).catch((e) => {
      if (my !== seq || !root.isConnected) return;
      setChildren(stage, h("div", { class: "callout callout-error", role: "alert" }, "图片加载失败：" + errorText(e)), info);
    });
  }

  function onKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (multi && e.key === "ArrowLeft") {
      e.preventDefault();
      show(cur - 1);
    } else if (multi && e.key === "ArrowRight") {
      e.preventDefault();
      show(cur + 1);
    } else if (e.key === "Tab") {
      const focusables = Array.prototype.filter.call(root.querySelectorAll("button, a[href]"), (el) => !el.disabled);
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (!root.contains(document.activeElement)) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  function close() {
    document.removeEventListener("keydown", onKey, true);
    root.remove();
    document.body.classList.remove("no-scroll");
    S.viewer = null;
    if (prevFocus && prevFocus.isConnected && typeof prevFocus.focus === "function") prevFocus.focus();
  }

  closeBtn.addEventListener("click", close);
  if (multi) {
    prevBtn.addEventListener("click", () => show(cur - 1));
    nextBtn.addEventListener("click", () => show(cur + 1));
  }
  root.addEventListener("click", (e) => {
    if (e.target === root || e.target === stage) close();
  });
  document.addEventListener("keydown", onKey, true);
  S.viewer = { close };
  document.body.appendChild(root);
  document.body.classList.add("no-scroll");
  closeBtn.focus();
  show(cur);
}

function showFatal(title, detail, retry) {
  const app = document.getElementById("app");
  app.setAttribute("aria-busy", "false");
  if (S.footer) {
    S.footer.el.remove();
    S.footer = null;
    document.body.classList.remove("has-footer");
  }
  setChildren(
    app,
    h(
      "div",
      { class: "fatal" },
      h(
        "div",
        { class: "fatal-card", role: "alert" },
        h("div", { class: "fatal-icon", "aria-hidden": "true" }, "⚠️"),
        h("h1", null, title),
        h("p", null, detail),
        retry ? h("button", { type: "button", class: "btn btn-primary", onclick: () => retry() }, icon("refresh"), "重试") : null,
      ),
    ),
  );
}

// ------------------------------------------------------------------ 表单字段

function fieldKind(field) {
  const type = field.type || "string";
  if (type === "bool") return "bool";
  if (type === "int" || type === "float") return "number";
  if (type === "list") return "list";
  if (type === "text") return "text";
  if (type !== "string") return "unsupported";
  if (field._special === "select_provider") return "provider";
  if (field._special === "select_persona") return "persona";
  if (field.secret) return "secret";
  if (Array.isArray(field.options) && field.options.length) return "options";
  return "string";
}

function listToText(v) {
  return Array.isArray(v) ? v.map(strOf).join("\n") : strOf(v);
}

function textToList(s) {
  return strOf(s)
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function buildSelect(id, choices, value, unknownSuffix, hintId) {
  const el = h("select", { id, class: "select", "aria-describedby": hintId || null });
  const set = (v) => {
    const val = strOf(v);
    const list = choices.slice();
    if (!list.some((c) => c.value === val)) list.push({ value: val, label: val ? val + unknownSuffix : "（空）" });
    setChildren(el, list.map((c) => h("option", { value: c.value }, c.label)));
    el.value = val;
  };
  set(value);
  return { el, set };
}

/**
 * 按 schema 渲染一个配置项。sec 为 null 时表示顶层字段（如 admin_ids）。
 * 返回控制器 { row, kind, field, input, set(v), apply(v), sync(), setDisabled(bool) }。
 */
function renderField(sec, key, field, opts) {
  opts = opts || {};
  const kind = fieldKind(field);
  const path = (sec ? sec + "." : "") + key;
  const id = uid("f-" + path);
  const hintId = field.hint ? id + "-hint" : "";
  const row = h("div", { class: "field field-" + kind + (opts.featured ? " field-featured" : "") });
  const ctl = { row, kind, key, sec, field, input: null, controls: [], disabled: false };
  let note = null; // (value) => { text, type } | null，不阻止保存的提示

  const current = () => {
    const v = getDraft(sec, key);
    return v === undefined ? clone(field.default) : v;
  };
  const value0 = current();

  const restoreBtn = h("button", { type: "button", class: "link-btn", hidden: true, title: "恢复为默认值" }, "恢复默认");
  const label = h(
    "label",
    { class: "field-label", for: id },
    opts.icon ? h("span", { class: "featured-icon" }, icon(opts.icon)) : null,
    field.description || key,
  );
  const head = h(
    "div",
    { class: "field-head" },
    h("div", { class: "field-title" }, label, h("code", { class: "field-key", "aria-hidden": "true" }, path)),
    restoreBtn,
  );
  const msg = h("p", { class: "field-msg", hidden: true });
  const hint = field.hint ? h("p", { class: "field-hint", id: hintId }, field.hint) : null;

  function sync() {
    const value = current();
    restoreBtn.hidden =
      ctl.disabled || kind === "secret" || kind === "unsupported" || field.default === undefined || deepEqual(value, field.default);
    const err = fieldError(field, value);
    if (ctl.input) ctl.input.classList.toggle("is-invalid", !!err);
    let m = err ? { text: err, type: "error" } : note ? note(value) : null;
    if (!m && ctl.disabled && opts.disabledNote) m = { text: opts.disabledNote, type: "info" };
    msg.hidden = !m;
    msg.className = "field-msg " + (m && m.type === "error" ? "is-error" : "is-info");
    msg.textContent = m ? m.text : "";
  }

  function commit(value) {
    setDraft(sec, key, value);
    sync();
  }

  let control = null;
  let extra = null;

  if (kind === "bool") {
    const btn = h("button", { type: "button", id, class: "switch", "aria-pressed": "false", "aria-describedby": hintId || null });
    const state = h("span", { class: "switch-state", "aria-hidden": "true" });
    const show = (v) => {
      btn.setAttribute("aria-pressed", v ? "true" : "false");
      state.textContent = v ? "开启" : "关闭";
    };
    show(asBool(value0));
    btn.addEventListener("click", () => {
      const v = btn.getAttribute("aria-pressed") !== "true";
      show(v);
      commit(v);
    });
    ctl.input = btn;
    ctl.controls = [btn];
    ctl.set = (v) => show(asBool(v));
    control = h("div", { class: "switch-wrap" }, state, btn);
  } else if (kind === "number") {
    const isInt = field.type === "int";
    const sl = isObj(field.slider) ? field.slider : null;
    const hasRange = !!sl && isFinite(Number(sl.min)) && isFinite(Number(sl.max)) && sl.min !== null && sl.max !== null;
    const step = sl && sl.step !== undefined && sl.step !== null ? sl.step : isInt ? 1 : "any";
    const parse = (raw) => {
      const s = strOf(raw).trim();
      if (s === "") return "";
      const n = Number(s);
      if (!isFinite(n) || (isInt && !Number.isInteger(n))) return s;
      return n;
    };
    const num = h("input", {
      type: "number",
      id,
      class: "input",
      inputmode: isInt ? "numeric" : "decimal",
      step,
      min: hasRange ? sl.min : null,
      max: hasRange ? sl.max : null,
      "aria-describedby": hintId || null,
    });
    num.value = strOf(value0);
    let range = null;
    if (hasRange) {
      range = h("input", {
        type: "range",
        class: "range",
        min: sl.min,
        max: sl.max,
        step,
        "aria-label": (field.description || key) + "（滑块）",
      });
      range.value = strOf(value0);
      range.addEventListener("input", () => {
        num.value = range.value;
        commit(parse(range.value));
      });
    }
    num.addEventListener("input", () => {
      const v = parse(num.value);
      if (range && typeof v === "number") range.value = String(v);
      commit(v);
    });
    if (hasRange) {
      note = (v) => {
        if (typeof v !== "number" || (v >= Number(sl.min) && v <= Number(sl.max))) return null;
        const bound = v < Number(sl.min) ? sl.min : sl.max;
        // 没有修改的值保存时原样保留（插件运行时仍按范围处理），修改后的值保存时会改为边界值
        const edited = !deepEqual(v, readCfg(S.saved, sec, key));
        return {
          text: "超出范围 " + sl.min + " ~ " + sl.max + (edited ? "，保存时会改为 " : "，使用时按 ") + bound +
            (edited ? "" : " 处理"),
          type: "info",
        };
      };
    }
    ctl.input = num;
    ctl.controls = range ? [num, range] : [num];
    ctl.set = (v) => {
      num.value = strOf(v);
      if (range) range.value = strOf(v);
    };
    control = h("div", { class: "num-wrap" }, num, range);
  } else if (kind === "text" || kind === "list" || kind === "unsupported") {
    const isList = kind === "list";
    const mono = kind === "unsupported" || (kind === "text" && !!field.editor_mode);
    const ta = h("textarea", {
      id,
      class: "textarea" + (mono ? " mono" : ""),
      rows: 3,
      spellcheck: mono || isList ? "false" : null,
      placeholder: isList ? "每行一项" : null,
      readOnly: kind === "unsupported",
      "aria-describedby": hintId || null,
    });
    const toText = (v) => (isList ? listToText(v) : kind === "unsupported" ? JSON.stringify(v, null, 2) || "" : strOf(v));
    ta.value = toText(value0);
    if (kind !== "unsupported") {
      ta.addEventListener("input", () => {
        autoGrow(ta);
        commit(isList ? textToList(ta.value) : ta.value);
      });
    }
    ctl.input = ta;
    ctl.controls = [ta];
    ctl.set = (v) => {
      ta.value = toText(v);
      autoGrow(ta);
    };
    control = ta;
    if (kind === "unsupported") {
      note = () => ({ text: "此类型的设置暂不支持在页面中修改，请到 AstrBot 的插件配置中修改。", type: "info" });
    } else if (kind === "text" && strOf(field.editor_language).toLowerCase() === "json") {
      note = (v) => {
        const s = strOf(v).trim();
        if (!s) return null;
        try {
          const parsed = JSON.parse(s);
          return isObj(parsed) ? null : { text: "提示：这里通常应填写 JSON 对象 {…}", type: "info" };
        } catch (e) {
          return { text: "JSON 格式有误：" + e.message, type: "error" };
        }
      };
      const fmt = h("button", { type: "button", class: "link-btn" }, "格式化 JSON");
      fmt.addEventListener("click", () => {
        try {
          const s = ta.value.trim() ? JSON.stringify(JSON.parse(ta.value), null, 2) : "";
          ta.value = s;
          autoGrow(ta);
          commit(s);
        } catch (e) {
          toast("JSON 格式有误，无法格式化", "error");
        }
      });
      ctl.controls.push(fmt);
      extra = h("div", { class: "json-tools" }, fmt);
    }
  } else if (kind === "secret") {
    const inp = h("input", {
      type: "password",
      id,
      class: "input",
      autocomplete: "new-password",
      spellcheck: "false",
      placeholder: "未设置",
      "aria-describedby": hintId || null,
    });
    inp.value = strOf(value0);
    const eye = h("button", { type: "button", class: "icon-btn", "aria-pressed": "false", "aria-label": "显示内容", title: "显示 / 隐藏" }, icon("eye"));
    eye.addEventListener("click", () => {
      const reveal = inp.type === "password";
      inp.type = reveal ? "text" : "password";
      eye.setAttribute("aria-pressed", reveal ? "true" : "false");
      eye.setAttribute("aria-label", reveal ? "隐藏内容" : "显示内容");
      setChildren(eye, icon(reveal ? "eyeOff" : "eye"));
    });
    inp.addEventListener("focus", () => {
      // 仍是掩码时全选，直接输入即可替换
      if (inp.value && inp.value.indexOf(S.secretMask) === 0) setTimeout(() => inp.select(), 0);
    });
    inp.addEventListener("input", () => commit(inp.value));
    note = (v) => {
      const saved = strOf(readCfg(S.saved, sec, key));
      const val = strOf(v);
      if (val && val === saved) return { text: "已保存（为安全起见不显示完整内容）。直接输入新值即可替换，清空后保存会删除。", type: "info" };
      if (!val && saved) return { text: "保存后会删除已保存的值。", type: "error" };
      if (val && S.secretMask && val.indexOf(S.secretMask.charAt(0)) !== -1) {
        return { text: "输入框里还留着掩码字符（•），请先清空再粘贴完整的值。", type: "error" };
      }
      if (val) return { text: "保存后将使用新输入的值。", type: "info" };
      return null;
    };
    ctl.input = inp;
    ctl.controls = [inp, eye];
    ctl.set = (v) => {
      inp.value = strOf(v);
    };
    control = h("div", { class: "secret-wrap" }, inp, eye);
  } else if (kind === "options" || kind === "provider" || kind === "persona") {
    let choices;
    let suffix;
    if (kind === "options") {
      const labels = Array.isArray(field.labels) ? field.labels : [];
      choices = field.options.map((o, i) => ({
        value: strOf(o),
        label: labels[i] !== undefined && labels[i] !== null && labels[i] !== "" ? strOf(labels[i]) : strOf(o),
      }));
      suffix = "（当前值，不在可选列表中）";
    } else if (kind === "provider") {
      choices = [{ value: "", label: EMPTY_LABELS[path] || FOLLOW_SESSION }].concat(
        S.providers.map((p) => ({ value: strOf(p.id), label: p.model ? strOf(p.id) + " (" + strOf(p.model) + ")" : strOf(p.id) })),
      );
      suffix = "（未找到此提供商）";
      if (!S.providers.length) note = () => ({ text: "没有获取到 LLM 提供商列表，请先在 AstrBot 中添加模型提供商。", type: "info" });
    } else {
      choices = [{ value: "", label: EMPTY_LABELS[path] || FOLLOW_SESSION }].concat(
        S.personas.map((p) => ({ value: strOf(p.name), label: strOf(p.label || p.name) })),
      );
      suffix = "（未找到此人格）";
    }
    const sel = buildSelect(id, choices, value0, suffix, hintId);
    sel.el.addEventListener("change", () => commit(sel.el.value));
    ctl.input = sel.el;
    ctl.controls = [sel.el];
    ctl.set = sel.set;
    control = sel.el;
  } else {
    const inp = h("input", {
      type: "text",
      id,
      class: "input" + (field.editor_mode ? " mono" : ""),
      spellcheck: "false",
      autocomplete: "off",
      placeholder: field.default ? strOf(field.default) : null,
      "aria-describedby": hintId || null,
    });
    inp.value = strOf(value0);
    inp.addEventListener("input", () => commit(inp.value));
    ctl.input = inp;
    ctl.controls = [inp];
    ctl.set = (v) => {
      inp.value = strOf(v);
    };
    control = inp;
  }

  ctl.sync = sync;
  ctl.apply = (v) => {
    ctl.set(v);
    commit(v);
  };
  ctl.setDisabled = (d) => {
    ctl.disabled = !!d;
    row.classList.toggle("is-disabled", ctl.disabled);
    ctl.controls.forEach((c) => {
      c.disabled = ctl.disabled;
    });
    sync();
  };
  restoreBtn.addEventListener("click", () => {
    ctl.apply(clone(field.default));
    if (ctl.input && typeof ctl.input.focus === "function") ctl.input.focus();
  });

  if (kind === "bool") append(row, [h("div", { class: "field-inline" }, head, control), hint, msg]);
  else append(row, [head, control, extra, hint, msg]);
  sync();
  return ctl;
}

/** 渲染一个配置分组（object 类型）的全部字段。 */
function renderSectionForm(sec) {
  const items = sectionItems(sec);
  const keys = Object.keys(items);
  const form = h("div", { class: "form" });
  const fields = {};
  const consumed = new Set();
  const featuredOf = (key) => {
    const fe = FEATURED[sec + "." + key];
    return fe && fieldKind(items[key]) === "bool" ? fe : null;
  };
  keys.forEach((key) => {
    const fe = featuredOf(key);
    if (!fe) return;
    fe.dependents.forEach((d) => {
      if (items[d] && d !== key) consumed.add(d);
    });
  });
  keys.forEach((key) => {
    if (consumed.has(key)) return;
    const fe = featuredOf(key);
    if (!fe) {
      const c = renderField(sec, key, items[key]);
      fields[key] = c;
      form.appendChild(c.row);
      return;
    }
    const main = renderField(sec, key, items[key], { featured: true, icon: fe.icon });
    fields[key] = main;
    const box = h("div", { class: "featured" }, main.row);
    const deps = fe.dependents
      .filter((d) => consumed.has(d))
      .map((d) => {
        const c = renderField(sec, d, items[d], { disabledNote: fe.disabledNote });
        fields[d] = c;
        box.appendChild(c.row);
        return c;
      });
    watch(box, () => {
      const on = asBool(getDraft(sec, key));
      deps.forEach((c) => {
        if (c.disabled === on) c.setDisabled(!on);
      });
    });
    form.appendChild(box);
  });
  return { el: form, fields };
}

/** 带标题和“未保存”标记的配置卡片。 */
function sectionCard(sec, opts) {
  opts = opts || {};
  const f = S.schema[sec];
  const title = opts.title || sectionTitle(sec);
  const dirty = h("span", { class: "badge badge-warn", hidden: true }, "未保存");
  const form = renderSectionForm(sec);
  const sub = f.hint ? h("p", { class: "card-sub" }, f.hint) : null;
  let el;
  if (opts.collapsible) {
    el = h("details", { class: "card settings-box" }, h("summary", null, title, " ", dirty), sub, form.el);
    if (opts.open) el.open = true;
    el.addEventListener("toggle", () => {
      if (opts.onToggle) opts.onToggle(el.open);
      if (el.open) growTextareas(el);
    });
  } else {
    el = h(
      "section",
      { class: "card" },
      h("div", { class: "card-head" }, h("div", null, h("h2", { class: "card-title" }, title), sub), dirty),
      form.el,
    );
  }
  watch(el, () => {
    dirty.hidden = deepEqual(S.draft[sec], S.saved[sec]);
  });
  return { el, fields: form.fields };
}

// ------------------------------------------------------------------ 绘图平台

function backendBadges(b) {
  const out = [];
  if (draftBackend() === b.key) out.push(h("span", { class: "badge badge-accent" }, "当前"));
  out.push(b.missing ? h("span", { class: "badge badge-warn", title: strOf(b.missing) }, "未配置") : h("span", { class: "badge badge-ok" }, "已配置"));
  return out;
}

function renderBackendsPanel(panel) {
  if (!S.backends.length) {
    panel.appendChild(h("div", { class: "empty" }, "没有可用的绘图平台"));
    return;
  }
  if (!findBackend(S.selected)) S.selected = S.backends[0].key;
  const list = h("div", { class: "bk-list", role: "group", "aria-label": "绘图平台" });
  S.backends.forEach((b) => {
    const badges = h("span", { class: "bk-badges" });
    const item = h(
      "button",
      {
        type: "button",
        class: "bk-item",
        "aria-current": b.key === S.selected ? "true" : "false",
        title: strOf(b.missing || b.summary) || null,
        onclick: () => selectBackend(b.key),
      },
      h("span", { class: "bk-name" }, strOf(b.label) || b.key),
      badges,
    );
    watch(item, () => setChildren(badges, backendBadges(b)));
    S.ui.bkItems.set(b.key, item);
    list.appendChild(item);
  });
  const main = h("div", { class: "bk-main" });
  S.ui.bkMain = main;
  panel.appendChild(h("div", { class: "bk-layout" }, list, main));
  renderBackendMain();
}

function selectBackend(key) {
  if (key === S.selected) return;
  S.selected = key;
  S.ui.bkItems.forEach((el, k) => el.setAttribute("aria-current", k === key ? "true" : "false"));
  renderBackendMain();
  const el = S.ui.bkItems.get(key);
  if (el && window.matchMedia && window.matchMedia("(max-width: 760px)").matches) {
    el.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
}

function renderBackendMain() {
  const main = S.ui.bkMain;
  const b = findBackend(S.selected);
  if (!main || !b) return;
  const dl = h("datalist", { id: uid("models-" + b.key) });
  S.ui.datalist = dl;
  S.ui.modelField = null;
  S.ui.models = null;
  S.ui.test = null;
  const formCard = backendFormCard(b, dl);
  const headCard = backendHeadCard(b);
  setChildren(main, headCard, formCard, testCard(b), dl);
  renderModels(b.key);
  renderTest(b.key);
  growTextareas(main);
}

function docRow(url) {
  if (!/^https?:\/\//i.test(strOf(url))) return null;
  // 部分 AstrBot 版本的 iframe 不允许打开新窗口，链接可能点不开，所以同时显示网址并提供复制
  const link = h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, "官方文档", icon("external"));
  const copy = h("button", { type: "button", class: "link-btn", title: "复制文档链接（如果点击链接没有反应）" }, "复制链接");
  copy.addEventListener("click", () => copyText(url).then((ok) => toast(ok ? "已复制链接" : "复制失败，请手动复制", ok ? "ok" : "error")));
  return h("div", { class: "doc-row" }, link, h("span", { class: "doc-url" }, url), copy);
}

function backendHeadCard(b) {
  const badges = h("div", { class: "bk-badges" });
  const mode = PROMPT_MODES[b.prompt_mode];
  watch(badges, () => setChildren(badges, backendBadges(b), mode ? h("span", { class: "badge", title: "提示词模式" }, mode) : null));
  const actions = h("div", { class: "card-actions" });
  const drawBackend = sectionItems("draw").backend;
  if (drawBackend) {
    const setBtn = h("button", { type: "button", class: "btn btn-primary btn-sm" });
    setBtn.addEventListener("click", () => {
      setDraft("draw", "backend", b.key);
      toast("已设为当前平台，保存后生效", "");
    });
    watch(setBtn, () => {
      const isCur = draftBackend() === b.key;
      setBtn.disabled = isCur;
      setChildren(setBtn, isCur ? "已是当前平台" : "设为当前平台");
    });
    actions.appendChild(setBtn);
  }
  let modelsBox = null;
  if (b.supports_models) {
    const btn = h("button", { type: "button", class: "btn btn-sm" });
    btn.addEventListener("click", () => fetchModels(b.key));
    modelsBox = h("div", { class: "models-box", "aria-live": "polite" });
    S.ui.models = { key: b.key, btn, box: modelsBox };
    actions.appendChild(btn);
  }
  return h(
    "section",
    { class: "card" },
    h(
      "div",
      { class: "card-head" },
      h("div", null, h("h2", { class: "card-title" }, strOf(b.label) || b.key), b.summary ? h("p", { class: "card-sub" }, strOf(b.summary)) : null),
      badges,
    ),
    b.missing ? h("div", { class: "callout callout-warn", role: "note" }, "⚠ ", strOf(b.missing), h("span", { class: "muted small" }, "（按已保存的设置检查）")) : null,
    actions.children.length ? actions : null,
    docRow(b.doc_url),
    modelsBox,
  );
}

function backendFormCard(b, dl) {
  if (!Object.keys(sectionItems(b.key)).length) {
    return h(
      "section",
      { class: "card" },
      h("h2", { class: "card-title" }, (strOf(b.label) || b.key) + " 设置"),
      h("div", { class: "callout callout-warn" }, "插件的配置文件（_conf_schema.json）中没有此平台的设置项（" + b.key + "）。"),
    );
  }
  const card = sectionCard(b.key);
  const mf = card.fields.model;
  if (mf && mf.kind === "string" && mf.input) mf.input.setAttribute("list", dl.id);
  S.ui.modelField = mf || null;
  return card.el;
}

function fetchModels(key) {
  const prev = S.models[key];
  const st = { loading: true, list: prev ? prev.list : [], error: "" };
  S.models[key] = st;
  renderModels(key);
  api("post", "models", { backend: key, section: clone(S.draft[key] || {}) })
    .then(
      (res) => {
        const seen = new Set();
        st.list = (Array.isArray(res.models) ? res.models : [])
          .map((m) => strOf(m).trim())
          .filter((m) => m && !seen.has(m) && seen.add(m));
        st.error = "";
        toast(st.list.length ? "获取到 " + st.list.length + " 个模型" : "平台没有返回可用的模型", st.list.length ? "ok" : "");
      },
      (e) => {
        st.error = e.message;
      },
    )
    .then(() => {
      st.loading = false;
      renderModels(key);
    });
}

function renderModels(key) {
  const ui = S.ui.models;
  if (!ui || ui.key !== key || !ui.box.isConnected) return;
  const st = S.models[key];
  const loading = !!(st && st.loading);
  ui.btn.disabled = loading;
  setChildren(ui.btn, loading ? [spinner("spinner-sm"), "获取中…"] : [icon("list"), st && st.list.length ? "重新获取模型列表" : "获取模型列表"]);
  if (S.ui.datalist) setChildren(S.ui.datalist, (st ? st.list : []).map((m) => h("option", { value: m })));
  setChildren(ui.box);
  if (!st) return;
  if (st.error) ui.box.appendChild(h("div", { class: "callout callout-error", role: "alert" }, st.error));
  if (!st.list.length) return;
  const mf = S.ui.modelField;
  ui.box.appendChild(
    h(
      "div",
      { class: "small muted" },
      "可用模型（共 " + st.list.length + " 个" + (mf ? "，点击填入「" + (mf.field.description || "model") + "」" : "") + "）",
    ),
  );
  const chips = h("div", { class: "model-chips" });
  const chipEls = st.list.map((m) => {
    const chip = h("button", { type: "button", class: "chip", title: m, "aria-pressed": "false" }, m);
    chip.addEventListener("click", () => pickModel(key, m));
    chips.appendChild(chip);
    return chip;
  });
  watch(chips, () => {
    const cur = strOf(getDraft(key, "model"));
    chipEls.forEach((c, i) => c.setAttribute("aria-pressed", st.list[i] === cur ? "true" : "false"));
  });
  ui.box.appendChild(chips);
}

function pickModel(key, model) {
  const mf = S.ui.modelField;
  if (!mf || mf.sec !== key) {
    copyText(model).then((ok) => toast(ok ? "此平台的设置中没有「model」字段，已复制模型名" : "此平台的设置中没有「model」字段", ok ? "" : "error"));
    return;
  }
  if (mf.kind === "options" && (mf.field.options || []).map(strOf).indexOf(model) < 0) {
    toast("「" + model + "」不在此平台的模型选项中，无法选择", "error");
    return;
  }
  mf.apply(model);
  toast("已填入模型：" + model, "ok");
}

function testCard(b) {
  const key = b.key;
  if (!S.tests[key]) {
    const def = strOf(getDraft("draw", "default_orientation"));
    S.tests[key] = {
      prompt: "",
      negative: "",
      orientation: ORIENTATIONS.some((o) => o[0] === def) ? def : "portrait",
      running: false,
      startedAt: 0,
      timer: 0,
      result: null,
      error: "",
    };
  }
  const t = S.tests[key];
  const sample = SAMPLE_PROMPTS[b.prompt_mode] || SAMPLE_PROMPTS.natural;
  const pid = uid("test-prompt-" + key);
  const nid = uid("test-negative-" + key);
  const oid = uid("test-orientation-" + key);

  const prompt = h("textarea", { id: pid, class: "textarea", rows: 3, placeholder: "例如：" + sample });
  prompt.value = t.prompt;
  prompt.addEventListener("input", () => {
    t.prompt = prompt.value;
    autoGrow(prompt);
  });
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runTest(key);
    }
  });

  let negative = null;
  if (b.supports_negative) {
    negative = h("textarea", { id: nid, class: "textarea", rows: 2, placeholder: "可选，例如：" + SAMPLE_NEGATIVE });
    negative.value = t.negative;
    negative.addEventListener("input", () => {
      t.negative = negative.value;
      autoGrow(negative);
    });
  }

  const fillSample = h("button", { type: "button", class: "link-btn" }, "填入示例");
  fillSample.addEventListener("click", () => {
    prompt.value = sample;
    t.prompt = sample;
    autoGrow(prompt);
    prompt.focus();
  });

  const orient = h("select", { id: oid, class: "select" }, ORIENTATIONS.map((o) => h("option", { value: o[0] }, o[1])));
  orient.value = t.orientation;
  orient.addEventListener("change", () => {
    t.orientation = orient.value;
  });

  const runBtn = h("button", { type: "button", class: "btn btn-primary" });
  runBtn.addEventListener("click", () => runTest(key));
  const status = h("div", { class: "test-status" });
  const result = h("div", { class: "test-out", "aria-live": "polite" });
  S.ui.test = { key, runBtn, status, result };

  return h(
    "section",
    { class: "card" },
    h(
      "div",
      { class: "card-head" },
      h("div", null, h("h2", { class: "card-title" }, "试画"), h("p", { class: "card-sub" }, "直接生成一张图，检查 Key、模型和尺寸等设置是否正确。")),
    ),
    h(
      "div",
      { class: "test-grid" },
      h(
        "div",
        null,
        h("div", { class: "field-head" }, h("label", { class: "field-label", for: pid }, "提示词"), fillSample),
        prompt,
      ),
      negative
        ? h("div", null, h("div", { class: "field-head" }, h("label", { class: "field-label", for: nid }, "负面提示词（可选）")), negative)
        : null,
      h(
        "div",
        { class: "test-row" },
        h("label", { class: "field-label", for: oid }, "画面方向"),
        orient,
        runBtn,
        status,
      ),
      h("p", { class: "test-note" }, TEST_NOTE),
      result,
    ),
  );
}

function updateTestStatus(key) {
  const ui = S.ui.test;
  if (!ui || ui.key !== key || !ui.status.isConnected) return;
  const t = S.tests[key];
  if (t.running) setChildren(ui.status, "已用时 " + fmtSec((Date.now() - t.startedAt) / 1000) + " 秒");
  else setChildren(ui.status);
}

function renderTest(key) {
  const ui = S.ui.test;
  if (!ui || ui.key !== key || !ui.result.isConnected) return;
  const t = S.tests[key];
  ui.runBtn.disabled = t.running;
  setChildren(ui.runBtn, t.running ? [spinner("spinner-sm"), "生成中…"] : [icon("play"), "试画"]);
  updateTestStatus(key);
  setChildren(ui.result);
  if (t.error) ui.result.appendChild(h("div", { class: "callout callout-error", role: "alert" }, t.error));
  if (t.result) ui.result.appendChild(testResultView(key, t.result));
}

function testResultView(key, r) {
  const b = findBackend(key);
  const title = (b ? strOf(b.label) : key) + " 试画";
  const imgBtn = h(
    "button",
    { type: "button", class: "test-image-btn", "aria-label": "查看大图", title: "点击查看大图" },
    h("img", { src: r.image, alt: "试画结果" }),
  );
  imgBtn.addEventListener("click", () =>
    openViewer([{ src: r.image, title, downloadName: r.downloadName, sizeBytes: dataUrlBytes(r.image) }], 0),
  );
  const chips = h("div", { class: "kv-chips" }, kv("用时", fmtSec(r.elapsed) + " 秒"));
  Object.keys(r.params).forEach((k) => chips.appendChild(kv(PARAM_LABELS[k] || k, fmtParam(r.params[k]), k)));
  return h(
    "div",
    { class: "test-result" },
    imgBtn,
    h(
      "div",
      { class: "test-info" },
      chips,
      r.text ? h("div", { class: "small muted test-text-label" }, "平台返回的文字") : null,
      r.text ? h("pre", { class: "pre" }, r.text) : null,
    ),
  );
}

function kv(k, v, title) {
  return h("span", { class: "kv", title: title || null }, h("span", { class: "kv-k" }, k), h("span", { class: "kv-v" }, v));
}

function runTest(key) {
  const t = S.tests[key];
  const b = findBackend(key);
  if (!t || t.running) return;
  if (!t.prompt.trim()) {
    t.error = "请先填写试画的提示词";
    t.result = null;
    renderTest(key);
    return;
  }
  t.running = true;
  t.error = "";
  t.result = null;
  t.startedAt = Date.now();
  clearInterval(t.timer);
  t.timer = setInterval(() => updateTestStatus(key), 100);
  renderTest(key);
  const body = {
    backend: key,
    section: clone(S.draft[key] || {}),
    prompt: t.prompt,
    negative: b && b.supports_negative ? t.negative : "",
    orientation: t.orientation,
  };
  api("post", "test", body)
    .then((res) => {
      if (!isImageDataUrl(res.image)) throw new Error("服务器没有返回有效的图片");
      const mime = strOf(res.mime) || mimeOf(res.image);
      t.result = {
        image: res.image,
        elapsed: isFinite(Number(res.elapsed)) ? Number(res.elapsed) : (Date.now() - t.startedAt) / 1000,
        params: isObj(res.params) ? res.params : {},
        text: strOf(res.text),
        downloadName: "multi_draw_" + key + "_" + stamp() + "." + extOf(mime),
      };
    })
    .catch((e) => {
      t.error = errorText(e);
    })
    .then(() => {
      t.running = false;
      clearInterval(t.timer);
      t.timer = 0;
      renderTest(key);
      if (S.tab !== "backends" || S.selected !== key) {
        const label = b ? strOf(b.label) : key;
        toast(t.error ? label + " 试画失败" : label + " 试画完成", t.error ? "error" : "ok");
      }
    });
}

// ------------------------------------------------------------------ 提示词与人格

function renderPromptsPanel(panel) {
  const names = PROMPT_SECTIONS.filter((k) => isSection(S.schema[k])).concat(
    Object.keys(S.schema).filter(
      (k) => isSection(S.schema[k]) && PROMPT_SECTIONS.indexOf(k) < 0 && RECORD_SECTIONS.indexOf(k) < 0 && !findBackend(k),
    ),
  );
  names.forEach((k) => panel.appendChild(sectionCard(k).el));

  const tops = Object.keys(S.schema).filter((k) => isObj(S.schema[k]) && !isSection(S.schema[k]));
  if (!tops.length) return;
  const form = h("div", { class: "form" });
  tops.forEach((k) => form.appendChild(renderField(null, k, S.schema[k]).row));
  const dirty = h("span", { class: "badge badge-warn", hidden: true }, "未保存");
  const title = tops.every((k) => k.indexOf("admin") >= 0) ? "管理员" : "其他设置";
  const card = h("section", { class: "card" }, h("div", { class: "card-head" }, h("h2", { class: "card-title" }, title), dirty), form);
  watch(card, () => {
    dirty.hidden = tops.every((k) => deepEqual(S.draft[k], S.saved[k]));
  });
  panel.appendChild(card);
}

// ------------------------------------------------------------------ 记录

function renderRecordsPanel(panel) {
  RECORD_SECTIONS.forEach((sec) => {
    if (!isSection(S.schema[sec])) return;
    const dirty = !deepEqual(S.draft[sec], S.saved[sec]);
    const card = sectionCard(sec, {
      collapsible: true,
      title: "记录设置",
      open: S.audit.settingsOpen || dirty,
      onToggle: (open) => {
        S.audit.settingsOpen = open;
      },
    });
    panel.appendChild(card.el);
  });
  const total = h("span", { class: "muted small" });
  const refresh = h("button", { type: "button", class: "btn btn-sm" }, icon("refresh"), "刷新");
  refresh.addEventListener("click", () => loadAudit(S.audit.page));
  const list = h("div", { class: "records" });
  const pager = h("nav", { class: "pager", "aria-label": "记录分页" });
  panel.appendChild(h("div", { class: "toolbar" }, h("h2", { class: "card-title" }, "绘图记录"), total, h("span", { class: "spacer" }), refresh));
  panel.appendChild(list);
  panel.appendChild(pager);
  S.ui.audit = { list, pager, total, refresh };
  if (!S.audit.data && !S.audit.loading) loadAudit(1);
  else renderAudit();
}

function loadAudit(page) {
  const A = S.audit;
  const my = ++A.seq;
  A.loading = true;
  A.error = "";
  renderAudit();
  api("get", "audit", { page, page_size: AUDIT_PAGE_SIZE })
    .then(
      (res) => {
        if (my !== A.seq) return;
        A.data = {
          records: Array.isArray(res.records) ? res.records.filter(isObj) : [],
          page: Number(res.page) || 1,
          pages: Number(res.pages) || 1,
          total: Number(res.total) || 0,
        };
        A.page = A.data.page;
      },
      (e) => {
        if (my === A.seq) A.error = e.message;
      },
    )
    .then(() => {
      if (my !== A.seq) return;
      A.loading = false;
      renderAudit();
    });
}

function renderAudit() {
  const ui = S.ui.audit;
  if (!ui || !ui.list.isConnected) return;
  const A = S.audit;
  const d = A.data;
  ui.refresh.disabled = A.loading;
  ui.total.textContent = d ? "共 " + d.total + " 条" : "";
  ui.list.setAttribute("aria-busy", A.loading ? "true" : "false");
  setChildren(ui.list);
  if (A.loading) ui.list.appendChild(h("div", { class: "test-status" }, spinner("spinner-sm"), "正在加载记录…"));
  if (A.error) {
    const retry = h("button", { type: "button", class: "link-btn" }, "重试");
    retry.addEventListener("click", () => loadAudit(A.page));
    ui.list.appendChild(h("div", { class: "callout callout-error", role: "alert" }, "加载失败：" + A.error + "  ", retry));
  }
  if (d && !A.loading) {
    if (!d.records.length) ui.list.appendChild(h("div", { class: "empty" }, "还没有绘图记录"));
    d.records.forEach((r) => ui.list.appendChild(recordCard(r)));
    fitClamps(ui.list);
  }
  setChildren(ui.pager);
  if (d && d.pages > 1) {
    const go = (p, text, disabled) => {
      const btn = h("button", { type: "button", class: "btn btn-sm", disabled: disabled || A.loading }, text);
      btn.addEventListener("click", () => {
        loadAudit(p);
        ui.list.scrollIntoView({ block: "start" });
      });
      return btn;
    };
    append(ui.pager, [
      go(1, "« 首页", d.page <= 1),
      go(d.page - 1, "‹ 上一页", d.page <= 1),
      h("span", { class: "pager-info" }, "第 " + d.page + " / " + d.pages + " 页"),
      go(d.page + 1, "下一页 ›", d.page >= d.pages),
      go(d.pages, "末页 »", d.page >= d.pages),
    ]);
  }
}

function addRecordText(grid, label, text, mono) {
  const s = strOf(text).trim();
  if (!s) return;
  const body = h("div", { class: "clamp-body is-clamped" + (mono ? " mono" : "") }, s);
  const toggle = h("button", { type: "button", class: "link-btn", "aria-expanded": "false" }, "展开");
  toggle.addEventListener("click", () => {
    const open = !body.classList.toggle("is-clamped");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.textContent = open ? "收起" : "展开";
  });
  append(grid, [h("div", { class: "rec-text-label" }, label), h("div", null, body, toggle)]);
}

/** 文字没有超过两行时去掉“展开”按钮。 */
function fitClamps(root) {
  Array.prototype.forEach.call(root.querySelectorAll(".clamp-body.is-clamped"), (body) => {
    if (body.scrollHeight <= body.clientHeight + 2) {
      body.classList.remove("is-clamped");
      const btn = body.nextElementSibling;
      if (btn) btn.remove();
    }
  });
}

function recordCard(r) {
  const ok = r.status === "success";
  const who = (strOf(r.user_name) || "?") + (r.user_id ? "（" + strOf(r.user_id) + "）" : "");
  const head = h(
    "div",
    { class: "rec-head" },
    h("span", { role: "img", "aria-label": ok ? "成功" : "失败", title: ok ? "成功" : "失败" }, ok ? "✅" : "❌"),
    r._no !== undefined && r._no !== null ? h("span", { class: "rec-no" }, "#" + strOf(r._no)) : null,
    h("span", { class: "rec-time" }, fmtTime(r.time)),
    h("span", { class: "rec-user" }, who),
    h("span", { class: "badge" }, r.group_id ? "群 " + strOf(r.group_id) : "私聊"),
    r.source ? h("span", { class: "badge" }, SOURCE_NAMES[r.source] || strOf(r.source)) : null,
  );
  const meta = h("div", { class: "rec-meta" });
  const add = (text, cls, title) => {
    if (text) meta.appendChild(h("span", { class: "badge" + (cls ? " " + cls : ""), title: title || null }, text));
  };
  add(strOf(r.backend_label || r.backend), "badge-accent", "绘图平台");
  add(strOf(r.model), "", "模型");
  add(strOf(r.size), "", "尺寸");
  if (r.steps) add(strOf(r.steps) + " 步", "", "采样步数");
  if (r.seed) add("seed " + strOf(r.seed), "", "种子");
  if (r.duration !== undefined && r.duration !== null) add("耗时 " + fmtSec(r.duration) + " 秒", "", "总耗时");
  if (r.anlas_cost !== undefined && r.anlas_cost !== null) {
    const cost = Number(r.anlas_cost);
    if (cost > 0) add("消耗 " + cost + " Anlas", "badge-warn", "Anlas 消耗");
    else if (cost === 0) add("0 Anlas（免费）", "badge-ok", "Anlas 消耗");
    else add("Anlas 变化 +" + Math.abs(cost), "", "生成期间余额增加了");
  } else if (r.opus_free === true) {
    add("预计免费", "badge-ok", "Opus 免费额度");
  } else if (r.opus_free === false) {
    add("可能消耗 Anlas", "badge-warn", "Opus 免费额度");
  }
  const card = h("article", { class: "rec" + (ok ? "" : " is-failed") }, head, meta.children.length ? meta : null);
  if (!ok) {
    const stage = r.stage ? "（" + (STAGE_NAMES[r.stage] || strOf(r.stage)) + "）" : "";
    card.appendChild(h("div", { class: "rec-error" }, "失败" + stage + "：" + (strOf(r.error) || "未知错误")));
  }
  const texts = h("div", { class: "rec-text" });
  addRecordText(texts, "需求", r.request, false);
  addRecordText(texts, "提示词", r.prompt, true);
  addRecordText(texts, "负面", r.negative, true);
  if (texts.children.length) card.appendChild(texts);
  if (r.image && typeof r.image === "string") {
    const name = r.image;
    const btn = h("button", { type: "button", class: "btn btn-sm" }, icon("image"), "查看图片");
    btn.addEventListener("click", () => openViewer([{ name, time: r.time, load: () => getFullImage(name) }], 0));
    card.appendChild(h("div", { class: "rec-actions" }, btn));
  }
  return card;
}

// ------------------------------------------------------------------ 图库

function getFullImage(name) {
  const G = S.gallery;
  if (G.full.has(name)) {
    const cached = G.full.get(name);
    G.full.delete(name);
    G.full.set(name, cached);
    return Promise.resolve(cached);
  }
  if (G.inflight.has(name)) return G.inflight.get(name);
  const p = api("get", "image", { name }).then((res) => {
    if (!isImageDataUrl(res.image)) throw new Error("服务器没有返回有效的图片");
    G.full.set(name, res.image);
    while (G.full.size > FULL_CACHE_SIZE) G.full.delete(G.full.keys().next().value);
    return res.image;
  });
  G.inflight.set(name, p);
  const done = () => G.inflight.delete(name);
  p.then(done, done);
  return p;
}

/** 用 canvas 缩成小图，避免图库同时持有几十张原图。失败时退回原图。 */
function makeThumb(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const w = img.naturalWidth;
        const hgt = img.naturalHeight;
        const scale = Math.min(1, THUMB_EDGE / Math.max(w, hgt, 1));
        if (scale >= 1) {
          resolve(dataUrl);
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(hgt * scale));
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      } catch (e) {
        resolve(dataUrl);
      }
    };
    img.onerror = () => reject(new Error("图片无法解码"));
    img.src = dataUrl;
  });
}

function enqueueThumb(name) {
  const G = S.gallery;
  if (!name || G.thumbs.has(name) || G.failed.has(name) || G.pending.has(name)) return;
  G.pending.add(name);
  G.queue.push(name);
  pumpThumbs();
}

function pumpThumbs() {
  const G = S.gallery;
  while (G.active < IMAGE_CONCURRENCY && G.queue.length) {
    const name = G.queue.shift();
    G.active += 1;
    getFullImage(name)
      .then(makeThumb)
      .then(
        (thumb) => {
          G.thumbs.set(name, thumb);
        },
        (e) => {
          G.failed.set(name, errorText(e));
        },
      )
      .then(() => {
        G.active -= 1;
        G.pending.delete(name);
        applyThumb(name);
        pumpThumbs();
      });
  }
}

function dropThumbQueue() {
  const G = S.gallery;
  G.queue.forEach((n) => G.pending.delete(n));
  G.queue = [];
}

function fillTile(tile, name) {
  const G = S.gallery;
  const cap = tile.querySelector(".tile-cap");
  setChildren(tile);
  if (G.thumbs.has(name)) {
    tile.appendChild(h("img", { src: G.thumbs.get(name), alt: "", decoding: "async" }));
  } else if (G.failed.has(name)) {
    tile.appendChild(h("span", { class: "tile-ph is-error", title: G.failed.get(name) }, "加载失败"));
  } else {
    tile.appendChild(h("span", { class: "tile-ph" }, "加载中…"));
  }
  if (cap) tile.appendChild(cap);
}

function applyThumb(name) {
  const tile = S.ui.tiles ? S.ui.tiles.get(name) : null;
  if (tile && tile.isConnected) fillTile(tile, name);
}

function galleryItems() {
  return (S.gallery.list || []).map((img) => ({
    name: img.name,
    time: img.time,
    sizeBytes: img.size_bytes,
    load: () => getFullImage(img.name),
  }));
}

function renderGalleryPanel(panel) {
  const count = h("span", { class: "muted small" });
  const refresh = h("button", { type: "button", class: "btn btn-sm" }, icon("refresh"), "刷新");
  refresh.addEventListener("click", loadGallery);
  const status = h("div", { "aria-live": "polite" });
  const grid = h("div", { class: "gallery" });
  panel.appendChild(h("div", { class: "toolbar" }, h("h2", { class: "card-title" }, "图库"), count, h("span", { class: "spacer" }), refresh));
  panel.appendChild(status);
  panel.appendChild(grid);
  S.ui.gallery = { grid, count, refresh, status };
  if (!S.gallery.list && !S.gallery.loading) loadGallery();
  else renderGallery();
}

function loadGallery() {
  const G = S.gallery;
  G.loading = true;
  G.error = "";
  G.failed.clear();
  renderGallery();
  api("get", "images", { limit: GALLERY_LIMIT })
    .then(
      (res) => {
        G.list = (Array.isArray(res.images) ? res.images : []).filter((x) => isObj(x) && typeof x.name === "string" && x.name);
        const names = new Set(G.list.map((x) => x.name));
        Array.from(G.thumbs.keys()).forEach((n) => {
          if (!names.has(n)) G.thumbs.delete(n);
        });
      },
      (e) => {
        G.error = e.message;
      },
    )
    .then(() => {
      G.loading = false;
      renderGallery();
    });
}

function renderGallery() {
  const ui = S.ui.gallery;
  if (!ui || !ui.grid.isConnected) return;
  const G = S.gallery;
  ui.refresh.disabled = G.loading;
  ui.count.textContent = G.list ? "最近 " + G.list.length + " 张" : "";
  setChildren(ui.status);
  if (G.loading) ui.status.appendChild(h("div", { class: "test-status" }, spinner("spinner-sm"), "正在加载图库…"));
  if (G.error) ui.status.appendChild(h("div", { class: "callout callout-error", role: "alert" }, "加载失败：" + G.error));
  if (S.ui.observer) S.ui.observer.disconnect();
  S.ui.observer = null;
  S.ui.tiles = new Map();
  setChildren(ui.grid);
  if (G.loading || !G.list) return;
  if (!G.list.length) {
    ui.status.appendChild(h("div", { class: "empty" }, "还没有保存的图片。开启「保存生成的图片」后，Bot 画的图会出现在这里。"));
    return;
  }
  const io =
    typeof IntersectionObserver === "function"
      ? new IntersectionObserver(
          (entries) => {
            entries.forEach((en) => {
              if (!en.isIntersecting) return;
              io.unobserve(en.target);
              enqueueThumb(en.target.getAttribute("data-name"));
            });
          },
          { rootMargin: "300px 0px" },
        )
      : null;
  S.ui.observer = io;
  G.list.forEach((img, i) => {
    const tile = h("button", {
      type: "button",
      class: "tile",
      "data-name": img.name,
      title: img.name,
      "aria-label": "查看大图：" + img.name,
    });
    tile.appendChild(h("span", { class: "tile-cap" }, fmtTime(img.time).slice(5, 16) || img.name));
    tile.addEventListener("click", () => openViewer(galleryItems(), i));
    fillTile(tile, img.name);
    S.ui.tiles.set(img.name, tile);
    ui.grid.appendChild(tile);
    if (!G.thumbs.has(img.name) && !G.failed.has(img.name)) {
      if (io) io.observe(tile);
      else enqueueThumb(img.name);
    }
  });
}

// ------------------------------------------------------------------ 页面框架

function renderHeader() {
  const current = h("span", { class: "badge badge-accent" });
  S.chrome.headerBackend = current;
  return h(
    "header",
    { class: "header" },
    h("div", { class: "logo", "aria-hidden": "true" }, icon("brush")),
    h(
      "div",
      { class: "title-wrap" },
      h("h1", { class: "title" }, "多平台绘图 · 绘图设置"),
      h("p", { class: "subtitle" }, "配置绘图平台与试画，查看绘图记录和图库"),
    ),
    h("div", { class: "header-badges" }, S.version ? h("span", { class: "badge", title: "插件版本" }, S.version) : null, current),
  );
}

function renderTabs() {
  const list = h("div", { class: "tabs", role: "tablist", "aria-label": "绘图设置分类" });
  S.chrome.tabs = {};
  S.chrome.tabDots = {};
  TABS.forEach((t) => {
    const dot = h("span", { class: "tab-dot", hidden: true, title: "有未保存的修改" }, h("span", { class: "sr-only" }, "（有未保存的修改）"));
    const btn = h(
      "button",
      {
        type: "button",
        class: "tab",
        role: "tab",
        id: "tab-" + t.key,
        "aria-controls": "panel",
        "aria-selected": t.key === S.tab ? "true" : "false",
        tabindex: t.key === S.tab ? "0" : "-1",
      },
      t.label,
      dot,
    );
    btn.addEventListener("click", () => switchTab(t.key));
    S.chrome.tabs[t.key] = btn;
    S.chrome.tabDots[t.key] = dot;
    list.appendChild(btn);
  });
  list.addEventListener("keydown", (e) => {
    const keys = TABS.map((t) => t.key);
    let i = keys.indexOf(S.tab);
    if (e.key === "ArrowRight") i = (i + 1) % keys.length;
    else if (e.key === "ArrowLeft") i = (i - 1 + keys.length) % keys.length;
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = keys.length - 1;
    else return;
    e.preventDefault();
    switchTab(keys[i]);
    S.chrome.tabs[keys[i]].focus();
  });
  return list;
}

function switchTab(key) {
  if (key === S.tab) return;
  S.tab = key;
  TABS.forEach((t) => {
    const btn = S.chrome.tabs[t.key];
    btn.setAttribute("aria-selected", t.key === key ? "true" : "false");
    btn.setAttribute("tabindex", t.key === key ? "0" : "-1");
  });
  renderPanel(false);
  const tabs = S.chrome.tabs[key].parentNode;
  if (tabs && window.scrollY > tabs.offsetTop) window.scrollTo(0, tabs.offsetTop);
}

function renderPanel(keepScroll) {
  const panel = S.chrome.panel;
  if (!panel) return;
  const y = window.scrollY;
  if (S.ui.observer) S.ui.observer.disconnect();
  if (S.tab !== "gallery") dropThumbQueue();
  S.ui = { bkItems: new Map(), tiles: new Map() };
  setChildren(panel);
  panel.setAttribute("aria-labelledby", "tab-" + S.tab);
  if (S.tab === "backends") renderBackendsPanel(panel);
  else if (S.tab === "prompts") renderPromptsPanel(panel);
  else if (S.tab === "records") renderRecordsPanel(panel);
  else renderGalleryPanel(panel);
  growTextareas(panel);
  if (keepScroll) window.scrollTo(0, y);
}

function ensureFooter() {
  if (S.footer) return;
  const detail = h("div", { class: "footer-detail" });
  const error = h("div", { class: "footer-error", role: "alert", hidden: true });
  const discard = h("button", { type: "button", class: "btn" }, "放弃");
  const save = h("button", { type: "button", class: "btn btn-primary", title: "保存（Ctrl+S）" }, "保存");
  discard.addEventListener("click", discardChanges);
  save.addEventListener("click", saveChanges);
  const el = h(
    "div",
    { class: "footer", role: "region", "aria-label": "未保存的修改" },
    h(
      "div",
      { class: "footer-inner" },
      h("div", { class: "footer-text" }, h("div", { class: "footer-title" }, "有未保存的修改"), detail, error),
      h("div", { class: "footer-actions" }, discard, save),
    ),
  );
  document.body.appendChild(el);
  S.footer = { el, detail, error, discard, save };
}

function refreshFooter(changed) {
  const f = S.footer;
  if (!f) return;
  const list = changed || changedSections();
  const open = list.length > 0 || S.saving;
  f.el.classList.toggle("is-open", open);
  f.el.inert = !open;
  if (open) f.el.removeAttribute("aria-hidden");
  else f.el.setAttribute("aria-hidden", "true");
  document.body.classList.toggle("has-footer", open);
  f.detail.textContent = list.length ? "修改了：" + list.map(sectionTitle).join("、") : "";
  f.error.hidden = !S.saveError;
  f.error.textContent = S.saveError;
  f.discard.disabled = S.saving;
  f.save.disabled = S.saving;
  setChildren(f.save, S.saving ? [spinner("spinner-sm"), "保存中…"] : "保存");
}

function refreshChrome() {
  const c = S.chrome;
  if (!c.headerBackend) return;
  const saved = savedBackend();
  const draft = draftBackend();
  c.headerBackend.textContent =
    "当前平台：" + backendLabel(saved) + (draft && draft !== saved ? "（未保存：" + backendLabel(draft) + "）" : "");
  const changed = changedSections();
  TABS.forEach((t) => {
    const dot = c.tabDots[t.key];
    if (dot) dot.hidden = !changed.some((k) => sectionTab(k) === t.key);
  });
  refreshFooter(changed);
}

function saveChanges() {
  if (S.saving) return;
  const changed = changedSections();
  if (!changed.length) return;
  const errors = draftErrors(changed);
  if (errors.length) {
    S.saveError = "请先修正以下设置：\n" + errors.join("\n");
    refreshFooter();
    toast("有设置填写不正确，未保存", "error");
    return;
  }
  const payload = {};
  changed.forEach((k) => {
    payload[k] = clone(S.draft[k]);
  });
  S.saving = true;
  S.saveError = "";
  refreshFooter();
  api("post", "config", { config: payload })
    .then((res) => {
      if (!isObj(res.config)) throw new Error("服务器返回的数据不完整");
      const before = S.saved;
      S.saved = res.config;
      // 保存期间又改过的分组保留草稿，其余换成服务器返回的值（敏感字段重新变成掩码）
      const draft = {};
      Object.keys(S.saved).forEach((k) => {
        const base = Object.prototype.hasOwnProperty.call(payload, k) ? payload[k] : before[k];
        draft[k] = deepEqual(S.draft[k], base) ? clone(S.saved[k]) : S.draft[k];
      });
      S.draft = draft;
      if (Array.isArray(res.backends)) S.backends = res.backends.filter((b) => isObj(b) && b.key);
      S.saving = false;
      renderPanel(true);
      refreshChrome();
      toast("已保存", "ok");
    })
    .catch((e) => {
      S.saving = false;
      S.saveError = errorText(e);
      refreshChrome();
      toast(S.saveError, "error");
    });
}

function discardChanges() {
  if (S.saving) return;
  S.draft = clone(S.saved);
  S.saveError = "";
  renderPanel(true);
  refreshChrome();
  if (S.chrome.panel) S.chrome.panel.focus({ preventScroll: true });
  toast("已放弃未保存的修改");
}

function renderApp() {
  const app = document.getElementById("app");
  app.setAttribute("aria-busy", "false");
  S.chrome = {};
  const header = renderHeader();
  const tabs = renderTabs();
  const panel = h("main", { id: "panel", class: "panel", role: "tabpanel", tabindex: "-1" });
  S.chrome.panel = panel;
  setChildren(app, header, tabs, panel);
  ensureFooter();
  renderPanel(false);
  refreshChrome();
}

function applyState(st) {
  S.version = strOf(st.version);
  S.schema = st.schema;
  S.saved = st.config;
  S.draft = clone(st.config);
  S.backends = Array.isArray(st.backends) ? st.backends.filter((b) => isObj(b) && b.key) : [];
  S.providers = Array.isArray(st.llm_providers) ? st.llm_providers.filter((p) => isObj(p) && p.id) : [];
  S.personas = Array.isArray(st.personas) ? st.personas.filter((p) => isObj(p) && p.name) : [];
  S.secretMask = strOf(st.secret_mask) || "••••";
  S.saveError = "";
  if (!findBackend(S.selected)) {
    const cur = savedBackend();
    S.selected = findBackend(cur) ? cur : S.backends.length ? S.backends[0].key : "";
  }
}

async function loadState() {
  const app = document.getElementById("app");
  app.setAttribute("aria-busy", "true");
  setChildren(app, h("div", { class: "boot" }, spinner(), h("span", null, "正在加载绘图设置…")));
  let st;
  try {
    st = await api("get", "state");
    if (!isObj(st.schema) || !isObj(st.config)) throw new Error("服务器返回的数据不完整");
  } catch (e) {
    showFatal(
      "无法加载绘图设置",
      errorText(e) + "\n\n请确认插件「多平台绘图」已启用；如果刚刚重载过插件，请稍后重试。",
      loadState,
    );
    return;
  }
  applyState(st);
  renderApp();
}

function onGlobalKey(e) {
  if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === "s" || e.key === "S")) {
    if (!S.footer || S.viewer) return;
    e.preventDefault();
    if (changedSections().length) saveChanges();
  }
}

async function init() {
  document.title = "绘图设置";
  window.addEventListener("keydown", onGlobalKey);
  window.addEventListener(
    "resize",
    debounce(() => growTextareas(document.body), 150),
  );
  const bridge = await waitForBridge();
  if (!bridge) {
    showFatal("未检测到 AstrBot 页面接口", "请在 AstrBot WebUI 的插件详情页中打开「绘图设置」页面。", () => window.location.reload());
    return;
  }
  S.bridge = bridge;
  let ctx = null;
  try {
    if (typeof bridge.ready === "function") ctx = await withTimeout(bridge.ready(), 15000, "等待 AstrBot 页面初始化超时");
  } catch (e) {
    showFatal("页面初始化失败", errorText(e), () => window.location.reload());
    return;
  }
  // AstrBot 4.24 的 bridge 只有 ready/getContext/apiGet/apiPost，上下文里也没有 isDark：
  // 此时保留服务器注入的 data-theme，没有的话 CSS 按 prefers-color-scheme 显示
  try {
    applyContextTheme(ctx);
    if (typeof bridge.getContext === "function") applyContextTheme(bridge.getContext());
    if (typeof bridge.onContext === "function") bridge.onContext(applyContextTheme);
  } catch (e) {
    // 忽略：主题仍由 data-theme / prefers-color-scheme 决定
  }
  await loadState();
}

init().catch((e) => {
  console.error("[multi_draw] 页面初始化失败", e);
  showFatal("页面出错了", errorText(e), () => window.location.reload());
});
