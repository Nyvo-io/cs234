# Course Attachment Router

在 `course` 总 vault 中，根据当前 Markdown 笔记的第一层目录动态设置附件路径：

- `cs234/...` -> `cs234/attachments`
- `cs229/...` -> `cs229/attachments`
- 根目录笔记 -> `attachments`

插件监听笔记打开、活动标签切换和编辑器粘贴事件。图片的实际写入、重名处理和 wikilink 插入仍由 Obsidian 完成。
