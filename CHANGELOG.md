# 更新日志

## v1.0.0（2026-10-25）

首个版本。基于 astrbot_plugin_nai_draw v1.2.0，参考 astrbot_plugin_angel_brush 加入多平台支持。

### 保留自 nai_draw

- LLM 函数工具（改名为 `draw_image`），Bot 在聊天中按需绘图，调用后主 LLM 不会重复回复。
- 开始绘图前按当前人格发一句开画提示；成功后发送图片和一句评价；失败时先发送错误信息，再发一句失落的话。三句台词的提示词可配置。
- NovelAI 的全部功能：Opus 免费参数保护、英文标签校验、Anlas 消耗记录、`/draw account` 查询余额。
- 只在确定请求没有被处理时自动重试，避免重复扣费。插件重载时会提示用户绘图被中断。

### 新增

- 新增 OpenAI、Seedream、xAI、Gemini、SenseNova、ComfyUI 六个绘图平台。每个平台都按官方文档实现，并且有独立的配置分组（Key、API 地址、代理、模型、各方向尺寸、平台特有参数、额外请求参数）。
  - Gemini 改用 `generateContent` 接口并正确设置 `imageConfig`，修正 angel_brush 中 Interactions 请求的 `mime_type: image/png` 导致的 400 错误，以及默认模型已停用的问题。
  - xAI 不再发送不被接受的 `size` 参数，改用宽高比和分辨率。
  - OpenAI 不再向 GPT Image 模型发送不支持的 `response_format`、`style`。
  - SenseNova 使用官方文档中的地址 `token.sensenova.cn` 和模型 `sensenova-u1.5-lite`。
  - ComfyUI 支持本地 / 自部署和 Comfy Cloud，内置 SDXL 工作流，支持在自定义工作流中使用占位符，超时或重载时会取消任务。
  - 填写的尺寸不符合模型要求时自动换成最接近的合法值。
- 「人格台词设置」新增「评价时附上生成的图片」开关和「看图评价使用的 LLM」：开启后把画好的图片一起发给多模态模型，让它看着图评价；失败时自动改为只根据提示词评价。
- 新增「绘图设置」页面（AstrBot ≥ 4.24.2）：可视化配置各平台和模型、获取模型列表、试画，查看绘图记录和图库。
- 提示词分为两种模式：NovelAI 和 ComfyUI 使用英文标签，其他平台使用自然语言画面描述。两种模式的系统提示词都可以修改。
- 生成的图片保存到本地图库，超过数量上限时自动删除最旧的图片。
- 超过大小阈值的图片会先转成 JPEG 再发送。
- 新增指令 `/draw status` 查看各平台配置情况，`/draw use <平台>` 切换绘图平台。原来的 `/nai account`、`/nai audit` 改为 `/draw account`、`/draw audit`。

### 改进

- 配置项、API Key 中误粘贴的换行、空格、`Bearer ` 前缀和多余的接口路径会被自动清理；Key 中含有非法字符时给出明确提示，错误信息中不会出现 Key。
- 访问本机或局域网地址（如本地 ComfyUI、自建的 NovelAI 中转）时不使用 AstrBot 的全局代理。
- 插件停用时会移除注册的页面接口。
- WebUI 对话中绘图超过 AstrBot 的工具调用超时时，不再丢失图片和评价：本轮回复先结束，之后完成的消息主动推送到页面或保存到对话历史。
- LLM 返回的 JSON 格式有误（如 `\(` 转义、多余的逗号、描述里没有转义的引号、JSON 后面还有其他文字）时也能正确取出提示词，不会把 JSON 原文当作提示词发给绘图平台。
- 修改「同时生成的图片数」后立即按新值生效，排队的请求按先后顺序开始。
- 生成失败的记录中也会包含模型、尺寸等参数（NovelAI 还包括步数、种子和是否免费），与 nai_draw 一致。
- 「指定人格」选择「默认人格」时使用 AstrBot 内置的默认人格。
- `/draw account` 中显示的 API 地址会隐藏其中的账号密码。
