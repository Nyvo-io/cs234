# Academic Term Lookup

这是一个直接安装在当前 vault 中的 Obsidian 插件。编辑 Markdown 时选中英文单词或短语，右键选择“查询……的中文释义”即可查看结果。

## AI 翻译记录

AI 翻译成功后，插件会把结果保存在当前笔记所属课程的隐藏目录中：

- `cs234/...` -> `cs234/.academic-term-lookup/ai-translations.json`
- `11-711 Advanced Natural Language Processing/...` -> `11-711 Advanced Natural Language Processing/.academic-term-lookup/ai-translations.json`

记录按照“课程内笔记路径 + 术语”隔离。同一术语在不同笔记中可以保留不同的上下文释义，不会跨课程混用。记录文件只包含术语、AI 返回内容和更新时间，不包含 API 密钥。

在编辑器的 Live Preview 中查询成功后，选中的术语会变成 `academic-term-lookup:` 链接；阅读视图也会为该笔记中已经记录的术语添加可点击样式。以后点击术语会直接打开保存的翻译弹窗，不会再次请求 AI。

只有结果来源为 **AI 学术术语翻译** 时才会写入课程记录。内置词典、离线词典和在线普通翻译保持一次性查询，不会生成持久记录或术语链接。插件升级前的内存查询结果不会迁移。

## 查询方式

- **内置双语词典**：无需密钥，优先查询离线词条，未命中时请求有道双语词典；网络失败时再使用通用翻译备用接口。
- **AI 专业翻译**：支持 OpenAI Chat Completions 和 Responses 两种兼容格式。在设置中填写基础地址或完整端点、模型和 API 密钥。插件会把选中词所在行作为上下文，并要求模型优先使用计算机科学、机器学习、数学和强化学习中的规范译法。结果按 Obsidian Markdown 渲染，支持行内和块级数学公式。AI 模式会直接使用 AI，只有“仅使用离线词典”会覆盖它。

## 离线词典

设置中的“离线词典优先”默认开启。插件内置一小份常见学术/强化学习术语，也支持两种扩展方式：

1. 在命令面板执行“导入离线词典”，粘贴 JSON 或 TSV，数据会保存到当前 vault 的插件设置中。
2. 把下载的词典文件放到 vault 中，在“离线词典文件路径”填入相对路径。JSON 可以是 `{"policy":["n. 政策；方针"]}`，TSV 每行格式为 `policy<Tab>n. 政策；方针`。

离线词典命中后不会访问网络；未命中才会继续在线查询。

设置中的“下载并安装”会下载开源 ECDICT 英汉词典（CSV），保存到插件私有目录 `.obsidian/plugins/academic-term-lookup/dictionaries/ecdict.csv`，然后建立本地词条索引。首次下载和加载需要一些时间与磁盘空间，之后可以开启“仅使用离线词典”完全断网使用。

DeepL 没有提供可下载的完整内部词库。DeepL API 是在线翻译服务，Glossary 只保存用户自己定义的词对，不能替代完整英汉词典。

## 隐私

开启“仅使用离线词典”时不会发出网络请求。在线词典模式只发送选中的词；AI 模式会把选中的词和所在行发送到你配置的 API 地址，请不要在上下文中放入不适合外传的内容。

## 安装

本目录已经位于当前 vault 的 `.obsidian/plugins/academic-term-lookup`。在 Obsidian 的“设置 → 社区插件”中刷新插件列表，启用 **Academic Term Lookup** 即可。
