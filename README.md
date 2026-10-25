# astrbot_plugin_multi_draw

让 AstrBot 借助 LLM 在聊天中画画的插件，支持 **NovelAI、OpenAI、Seedream、xAI、Gemini、SenseNova、ComfyUI** 七个绘图平台，兼容 OneBot11（aiocqhttp，如 NapCat、Lagrange）。

在 [astrbot_plugin_nai_draw](https://github.com/chenxuxin188/astrbot_plugin_nai_draw) 的基础上，参考 [astrbot_plugin_angel_brush](https://github.com/kawayiYokami/astrbot_plugin_angel_brush) 加入了多平台支持和可视化配置页面。各平台的接口都按官方文档重新实现，修正了 angel_brush 中 Gemini 因 MIME 类型错误无法使用等问题。

## 功能

- **对话中绘图**：插件向 Bot 的 LLM 注册了 `draw_image` 函数工具。用户说「画一只在樱花树下睡觉的猫娘」之类的话时，Bot 会自己决定调用它。
- **按人格说话**：
  - 开始绘图时，按 Bot 当前人格的口吻发一句明确的「正在画」提示。
  - 成功后发送图片，再用人格口吻简短评价一下。可以选择**把画好的图片一起发给 LLM**，让它看着图评价。
  - 失败时先把错误信息发出来，再说一句失落的话。
  - 这三句话的提示词都可以在配置中修改。
- **七个绘图平台**：在配置或「绘图设置」页面中填好 Key，选择当前使用的平台即可；管理员也可以用 `/draw use <平台>` 随时切换。
- **LLM 生成提示词**：用指定的 LLM 把自然语言需求转换成绘图提示词，并自动选择竖图、横图或方图。
  - NovelAI（以及 ComfyUI 的标签模式）使用 Danbooru 风格的英文标签。
  - 其他平台使用一段详细的英文画面描述（需要画进图里的文字保留原文）。
- **绘图设置页面**：在 AstrBot WebUI 中可视化配置各平台和模型、获取模型列表、试画，查看绘图记录和图库。
- **绘图记录与图库**：每次绘图都会记录平台、模型、尺寸、耗时、提示词和失败原因；生成的图片保存在本地。

## 支持的平台

| 平台 | 默认模型 | 提示词 | 需要准备 |
| --- | --- | --- | --- |
| NovelAI | `nai-diffusion-4-5-full` | 英文标签 | Persistent API Token（`pst-` 开头）。默认使用 Opus 免费参数 |
| OpenAI（GPT Image） | `gpt-image-2.5-flare` | 自然语言 | API Key；可能需要先完成组织验证。也支持 OpenAI 兼容的中转服务 |
| Seedream（火山方舟 / 豆包） | `doubao-seedream-5-0-pro-260628` | 自然语言 | 方舟 API Key，并在控制台开通模型。也支持海外 BytePlus ModelArk |
| xAI（Grok Imagine） | `grok-imagine-image-2.0` | 自然语言 | xAI API Key（`xai-` 开头），账户需要先充值 |
| Gemini（Nano Banana） | `gemini-nano-banana-2.1` | 自然语言 | Google AI Studio API Key；免费层可能没有图片模型额度 |
| SenseNova（商汤日日新） | `sensenova-u1.5-lite` | 自然语言 | SenseNova 开放平台 API Key（`sk-` 开头），国内站 / 国际站地址需与 Key 对应 |
| ComfyUI | 模型列表中的第一个 | 英文标签（可改为自然语言） | 本地或自部署的 ComfyUI，或 Comfy Cloud（需要 API Key） |

各平台的尺寸、画质、水印等参数都可以在配置中调整，配置项的说明（问号图标）里写明了可选值和获取 Key 的位置。填写的尺寸不符合所选模型的要求时，插件会自动换成最接近的合法值，不会直接报错。

## 安装

在 AstrBot WebUI 的插件市场中通过仓库地址安装：

```
https://github.com/chenxuxin188/astrbot_plugin_multi_draw
```

也可以把仓库克隆到 `data/plugins/` 目录后重启 AstrBot。要求 AstrBot ≥ 4.10；「绘图设置」页面要求 AstrBot ≥ 4.24.2。

## 配置

可以在 WebUI 的插件配置页面中填写，也可以在「绘图设置」页面中配置（见下一节）。

| 配置分组 | 说明 |
| --- | --- |
| 管理员 ID 列表 | 可以使用管理员指令的 QQ 号 |
| 通用绘图设置 | 当前使用的绘图平台、默认画面方向、同时生成的图片数、是否保存图片、最多保留图片数、发送前压缩的大小阈值 |
| NovelAI / OpenAI / Seedream / xAI / Gemini / SenseNova / ComfyUI 设置 | 各平台的 Key、API 地址、代理、模型、竖图 / 横图 / 方图尺寸和平台特有的参数，以及「额外请求参数」（JSON，会合并进请求体，供高级用户覆盖参数） |
| 提示词设置 | 生成提示词使用的 LLM、标签模式和自然语言模式的系统提示词、仅生成全年龄内容、额外要求 |
| 人格台词设置 | 生成台词使用的 LLM、指定人格、台词长度、三句台词的提示词、**评价时附上生成的图片**、看图评价使用的 LLM |
| 绘图记录 | 是否记录 NovelAI 的 Anlas 消耗、每页条数、最多保留条数 |

### 评价时附上生成的图片

「人格台词设置 → 评价时附上生成的图片」默认关闭。

- **关闭**：评价时 LLM 只能看到绘图需求和提示词，靠想象评价。
- **开启**：图片发出后，插件把画好的图片缩小成 JPEG（长边 1024），和评价提示词一起发给「看图评价使用的 LLM」（留空则使用「生成人格台词使用的 LLM」或当前会话的模型），让它看着图评价。
  - 需要支持图片输入的多模态模型。提供商设置中明确声明不支持图片的模型会被跳过。
  - 看图评价失败（模型不支持图片、超时等）时，自动改为只根据提示词评价，不会影响出图。

## 绘图设置页面

AstrBot ≥ 4.24.2 中，在 WebUI 的插件页面点开本插件，进入「绘图设置」页面（4.29 起也可以在侧边栏「扩展功能」中找到）。

- **绘图平台**：查看七个平台是否已配置，编辑各平台的设置，设为当前平台，获取模型列表（OpenAI、xAI、Gemini、SenseNova、ComfyUI），打开官方文档。
  - **试画**：用页面上的设置（包括尚未保存的修改）直接把提示词发给绘图平台，检查 Key、模型和尺寸是否可用。试画不经过 LLM，会正常计费。
- **提示词与人格**：编辑通用绘图设置、提示词设置、人格台词设置和管理员列表。LLM 和人格可以从下拉列表中选择。
- **记录**：分页查看绘图记录。
- **图库**：查看最近生成的图片，点击放大或下载。

页面上的 API Key 只显示末 4 位；不修改时保存不会改变原来的 Key，清空后保存会删除。在页面中保存配置会立即生效，不会重载插件，正在进行的绘图不受影响。

## 使用

### 普通用户

- 直接和 Bot 说想画什么，例如「帮我画一张初音未来在海边的横图」。群聊里需要按 AstrBot 的唤醒规则 @Bot 或使用唤醒前缀。

### 指令

| 指令 | 作用 |
| --- | --- |
| `/draw status` | 查看当前绘图平台，以及各平台是否已配置 |
| `/draw use <平台>`（管理员） | 切换绘图平台，例如 `/draw use openai`。也接受 `nai`、`gpt`、`豆包`、`grok`、`banana`、`商汤`、`comfy` 等别名 |
| `/draw account`（管理员） | 查询 NovelAI 的订阅等级、到期时间、Anlas 余额，以及当前参数是否免费 |
| `/draw audit [页码]`（管理员） | 从新到旧查看绘图记录：时间、用户、群号、平台、模型、尺寸、耗时、Anlas 消耗、需求和提示词、失败原因 |

数据保存在 `data/plugin_data/astrbot_plugin_multi_draw/`：绘图记录为 `audit.jsonl`，图片在 `images/` 目录下按日期存放。

## 注意事项

- **Bot 不调用绘图工具时，请检查：**
  - 当前使用的模型支持函数调用（Function Calling）。
  - WebUI 中 `draw_image` 工具处于启用状态。
  - 如果人格设置了工具白名单，需要把 `draw_image` 加进去。
- **从 nai_draw 迁移**：两个插件可以同时安装，但请只启用一个，否则 LLM 会看到 `nai_draw` 和 `draw_image` 两个绘图工具。NovelAI 的设置项基本相同（「必定包含的正面 / 负面提示词」移到了 NovelAI 设置中，默认画面方向移到了通用绘图设置中），可以照着原来的配置填写。
- **重复回复**：工具调用后，主 LLM 不会再额外回复。但个别模型会在调用工具的同时输出一段文字，这段文字也会发出。
- **日志中的 `draw_image 没有返回值，或者已将结果直接发送给用户。`**：属于正常现象。插件让工具返回空值来结束本轮对话，避免 Bot 重复回复。
- **排队与并发**：同一个平台同时进行的请求数由「同时生成的图片数」控制，超出的会排队。NovelAI 固定一次一张（同一账号同时只能生成一张）。同一用户在上一张完成前不能发起新的请求。
- **失败重试**：只有确定请求没有被处理时才会自动重试一次，例如连接失败、限流、网关暂时不可用。超时、可能已经开始生成的服务器错误、额度用尽都不会重试，避免重复扣费。
- **代理**：每个平台都可以单独设置代理（http / https / socks5）；留空时使用 AstrBot 系统设置中的全局代理。访问本机、局域网、Docker 服务名等地址（如本地 ComfyUI）时不使用全局代理。
- **ComfyUI**：
  - 留空「自定义工作流」时使用内置的 SDXL 文生图工作流，适用于 SDXL / Illustrious / NoobAI / Pony 等模型。
  - 使用自己的工作流时，请在 ComfyUI 中用「导出 (API)」导出 API 格式的 JSON，可以使用 `{{prompt}}`、`{{negative}}`、`{{seed}}`、`{{width}}`、`{{height}}` 等占位符，详见配置说明。
  - 生成超时或插件重载时，会尝试取消 ComfyUI 中对应的任务。
  - AstrBot 在 Docker 中运行时，`127.0.0.1` 指容器自身，请填写 `http://host.docker.internal:8188` 或局域网 IP，并用 `--listen` 启动 ComfyUI。
- **Gemini**：默认使用 `generateContent` 接口；模型不在该接口上提供时自动改用 Interactions API。已停用的 `gemini-3-pro-image-preview` 等预览模型会自动换成官方替代模型。
- **xAI**：接口不接受像素尺寸，尺寸设置使用宽高比（如 `2:3`），也可以填像素尺寸，插件会换算成最接近的宽高比和分辨率。
- **大图发送**：图片超过「发送前压缩的大小阈值」（默认 8 MB）时会先转成 JPEG 再发送，图库中保存的仍是原图。
- **重载插件**：在 WebUI 的插件配置页面保存会触发重载，正在进行的绘图会被中断，Bot 会提示用户稍后重试。在「绘图设置」页面保存不会重载插件。两个页面同时打开时，以最后保存的为准。
- **WebUI 对话**：在 WebUI 自带的聊天页面中，Bot 会尽量等绘图完成后再结束本轮回复，但最多等到 AstrBot 的工具调用超时（`tool_call_timeout`，新版本默认 120 秒，较早的版本和旧配置为 60 秒）前 10 秒。绘图更慢时本轮回复会先结束，之后完成的图片和评价改为主动发送：新版本 AstrBot 会推送到页面或保存到对话历史中，部分旧版本（约 4.12–4.18）可能无法显示。绘图经常较慢时，建议在 AstrBot 设置中把工具调用超时调大到超过绘图平台的超时时间。

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。

## 开发

```bash
python -m pip install httpx pytest pillow
python -m pytest tests
```

- `backends/`、`prompting.py`、`audit.py`、`gallery.py`、`settings.py`、`imaging.py` 不依赖 AstrBot，可以单独测试。
- `tests/test_main_flow.py` 用 `tests/astrbot_stub.py` 模拟 AstrBot，端到端测试 `main.py` 和页面接口。
- 所有测试的 HTTP 请求都由 `httpx.MockTransport` 模拟，不会访问网络。
- 「绘图设置」页面在 `pages/draw-settings/`，是不需要构建的原生 JavaScript；表单根据 `_conf_schema.json` 自动生成。
