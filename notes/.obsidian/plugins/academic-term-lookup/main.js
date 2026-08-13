const fs = require("fs");
const path = require("path");

const {
  MarkdownRenderer,
  Modal,
  Menu,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  requestUrl,
} = require("obsidian");

const DEFAULT_SETTINGS = {
  provider: "dictionary",
  preferOffline: true,
  offlineOnly: false,
  offlineDictionaryPath: "",
  offlineDictionaryDownloadUrl: "https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.csv",
  aiFormat: "chat",
  aiEndpoint: "https://api.openai.com/v1/chat/completions",
  aiModel: "gpt-4o-mini",
  aiApiKey: "",
  includeContext: true,
};

const ECDICT_PATH = ".obsidian/plugins/academic-term-lookup/dictionaries/ecdict.csv";
const AI_TRANSLATION_FOLDER = ".academic-term-lookup";
const AI_TRANSLATION_FILE = "ai-translations.json";
const TERM_LINK_PREFIX = "academic-term-lookup:";

const BUILTIN_DICTIONARY = {
  tabular: ["adj. 表格式的；表格型的（如 tabular MDP）"],
  mdp: ["n. 马尔可夫决策过程（Markov Decision Process）"],
  "tabular mdp": ["n. 表格式马尔可夫决策过程；以表格表示状态、动作和转移的 MDP"],
  "policy iteration": ["n. 策略迭代（通过策略评估与策略改进求解最优策略）"],
  "value iteration": ["n. 价值迭代（通过 Bellman 最优算子迭代求解最优价值函数）"],
  "optimal policy": ["n. 最优策略"],
  greedy: ["adj. 贪心的；每一步选择当前最优动作的"],
  backup: ["n. 回溯更新；备份操作（强化学习中的价值更新）", "v. 回溯更新；备份"],
  policy: ["n. 政策；方针；策略"],
  iteration: ["n. 迭代；反复；迭代过程"],
  optimization: ["n. 优化；最优化"],
  objective: ["n. 目标函数；目标；客观事物", "adj. 客观的"],
  gradient: ["n. 梯度；渐变；坡度"],
  embedding: ["n. 嵌入；嵌入表示（机器学习）"],
  attention: ["n. 注意力机制；注意；关注"],
  representation: ["n. 表示；表征；表示形式"],
  inference: ["n. 推理；推断；模型推断"],
  exploration: ["n. 探索（强化学习中的探索行为）"],
  exploitation: ["n. 利用（强化学习中的利用已知信息）"],
  stochastic: ["adj. 随机的；随机过程的"],
  convergence: ["n. 收敛；收敛性"],
  trajectory: ["n. 轨迹；状态-动作轨迹"],
  rollout: ["n. 展开；采样轨迹；（策略） rollout"],
  reward: ["n. 奖励；回报（强化学习）", "v. 奖励"],
  value: ["n. 价值；价值函数（强化学习）", "v. 评估"],
  "value function": ["n. 价值函数（从某状态或状态-动作对出发的期望回报）"],
  state: ["n. 状态；状态空间", "v. 陈述；说明"],
  action: ["n. 动作；行动；动作空间", "v. 采取行动"],
  sampling: ["n. 采样；抽样"],
  likelihood: ["n. 似然；似然函数"],
  posterior: ["n. 后验分布；后验概率", "adj. 后验的"],
  prior: ["n. 先验分布；先验概率", "adj. 先验的"],
};

function normalizeTerm(value) {
  return value
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[\s"'“”‘’`]+|[\s"'“”‘’`.,;:!?，。；：！？]+$/g, "");
}

function courseRootForPath(filePath, vaultIsCourseRoot = false) {
  const normalizedPath = String(filePath || "");
  if (!normalizedPath || normalizedPath.startsWith(".")) return null;
  if (vaultIsCourseRoot) return "";
  const [root, ...rest] = normalizedPath.split("/");
  if (!root || rest.length === 0 || root.startsWith(".")) return null;
  return root;
}

function courseTranslationPath(courseRoot) {
  return [courseRoot, AI_TRANSLATION_FOLDER, AI_TRANSLATION_FILE].filter(Boolean).join("/");
}

function courseRelativePath(filePath, courseRoot) {
  if (courseRoot === "") return String(filePath || "") || null;
  const prefix = `${courseRoot}/`;
  return String(filePath || "").startsWith(prefix)
    ? String(filePath).slice(prefix.length)
    : null;
}

function resolveCourseStorageScope(vaultBasePath, notePath) {
  const basePath = String(vaultBasePath || "");
  const relativePath = String(notePath || "");
  if (!basePath || !relativePath || relativePath.startsWith(".")) return null;

  const absoluteNotePath = path.resolve(basePath, relativePath);
  const segments = absoluteNotePath.split(path.sep);
  const libraryIndex = segments.lastIndexOf("course");
  if (libraryIndex < 0 || libraryIndex >= segments.length - 1) return null;

  const libraryPath = segments.slice(0, libraryIndex + 1).join(path.sep) || path.parse(path.sep).root;
  const courseSegments = segments.slice(libraryIndex + 1);
  const courseRoot = courseSegments.length > 1 ? courseSegments.shift() : "";
  const canonicalNotePath = courseSegments.join("/");
  if (!canonicalNotePath) return null;

  const coursePath = courseRoot ? path.join(libraryPath, courseRoot) : libraryPath;
  const folderPath = path.join(coursePath, AI_TRANSLATION_FOLDER);
  return {
    courseRoot,
    notePath: canonicalNotePath,
    filePath: path.join(folderPath, AI_TRANSLATION_FILE),
    folderPath,
    absolute: true,
  };
}

function normalizeAIRecords(value) {
  const notes = value && typeof value.notes === "object" && !Array.isArray(value.notes)
    ? value.notes
    : {};
  return { version: 1, notes };
}

function recordAITranslation(value, notePath, rawTerm, result, updatedAt = new Date().toISOString()) {
  const records = normalizeAIRecords(value);
  const term = normalizeTerm(rawTerm);
  if (!notePath || !term || result?.kind !== "ai") return records;
  const previousNote = records.notes[notePath] && typeof records.notes[notePath] === "object"
    ? records.notes[notePath]
    : {};
  records.notes[notePath] = {
    ...previousNote,
    [term.toLowerCase()]: { term, result, updatedAt },
  };
  return records;
}

function createTermLink(term, label = term) {
  const normalized = normalizeTerm(term).toLowerCase();
  const rawLabel = String(label);
  const leading = rawLabel.match(/^\s*/)?.[0] || "";
  const trailing = rawLabel.match(/\s*$/)?.[0] || "";
  const displayLabel = rawLabel.slice(leading.length, rawLabel.length - trailing.length || undefined);
  return `${leading}[${displayLabel}](${TERM_LINK_PREFIX}${encodeURIComponent(normalized)})${trailing}`;
}

function parseTermLink(href) {
  const value = String(href || "");
  if (!value.startsWith(TERM_LINK_PREFIX)) return null;
  try {
    const term = normalizeTerm(decodeURIComponent(value.slice(TERM_LINK_PREFIX.length))).toLowerCase();
    return term || null;
  } catch (error) {
    return null;
  }
}

function cachedLinkAtPosition(line, ch) {
  const pattern = /\[([^\]]+)\]\((academic-term-lookup:[^)]+)\)/g;
  const source = String(line || "");
  let match;
  while ((match = pattern.exec(source))) {
    if (ch < match.index || ch > match.index + match[0].length) continue;
    const term = parseTermLink(match[2]);
    if (term) return { term, label: normalizeTerm(match[1]) };
  }
  return null;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function termMatcher(terms) {
  const alternatives = [...new Set(terms.map((term) => normalizeTerm(term).toLowerCase()).filter(Boolean))]
    .sort((left, right) => right.length - left.length)
    .map(escapeRegExp);
  return alternatives.length ? new RegExp(`\\b(${alternatives.join("|")})\\b`, "gi") : null;
}

function looksLikeEnglish(value) {
  return /[A-Za-z]/.test(value) && /^[A-Za-z0-9][A-Za-z0-9\s'’\-_/().]*$/.test(value);
}

function stripHtml(value) {
  const holder = document.createElement("div");
  holder.innerHTML = String(value || "");
  return (holder.textContent || holder.innerText || "").replace(/\s+/g, " ").trim();
}

function cleanDictionaryField(value) {
  const text = String(value || "");
  return /<[^>]+>/.test(text)
    ? stripHtml(text)
    : text.replace(/\s+/g, " ").trim();
}

function extractDictionaryText(value) {
  if (typeof value === "string") return stripHtml(value);
  if (Array.isArray(value)) return unique(value.map(extractDictionaryText)).join("；");
  if (!value || typeof value !== "object") return "";
  for (const key of ["i", "l", "text", "t", "value", "content", "translation"]) {
    const text = extractDictionaryText(value[key]);
    if (text) return text;
  }
  return "";
}

function unique(values) {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function resolveAIEndpoint(endpoint, format) {
  const value = String(endpoint || "").trim().replace(/\/+$/, "");
  if (!value) return "";
  if (format === "responses") {
    if (/\/responses$/i.test(value)) return value;
    if (/\/v1$/i.test(value)) return `${value}/responses`;
    return `${value}/v1/responses`;
  }
  if (/\/chat\/completions$/i.test(value)) return value;
  if (/\/v1$/i.test(value)) return `${value}/chat/completions`;
  return `${value}/v1/chat/completions`;
}

function extractAIContent(data) {
  const chatContent = data?.choices?.[0]?.message?.content;
  if (typeof chatContent === "string" && chatContent.trim()) return chatContent.trim();
  if (typeof data?.output_text === "string" && data.output_text.trim()) return data.output_text.trim();
  if (typeof data?.output === "string" && data.output.trim()) return data.output.trim();

  const parts = [];
  for (const item of Array.isArray(data?.output) ? data.output : []) {
    if (typeof item === "string") {
      parts.push(item);
      continue;
    }
    for (const content of Array.isArray(item?.content) ? item.content : []) {
      const text = typeof content === "string"
        ? content
        : content?.text || content?.output_text || content?.value;
      if (typeof text === "string" && text.trim()) parts.push(text.trim());
    }
  }
  return parts.join("\n").trim();
}

function parseYoudao(data, term) {
  const rawWords = data?.ec?.word || data?.simple?.word || [];
  const words = Array.isArray(rawWords) ? rawWords : [rawWords];
  const word = words[0] || {};
  const meanings = [];
  const translations = Array.isArray(word.trs) ? word.trs : [word.trs].filter(Boolean);
  for (const item of translations) {
    const candidates = Array.isArray(item.tr) ? item.tr : [item.tr || item];
    for (const candidate of candidates) {
      const text = extractDictionaryText(candidate);
      if (text) meanings.push(text);
    }
  }
  const result = {
    kind: "dictionary",
    source: "有道双语词典",
    term,
    meanings: unique(meanings),
    pronunciation: [word.ukphone && `英 /${word.ukphone}/`, word.usphone && `美 /${word.usphone}/`]
      .filter(Boolean)
      .join("  "),
    examples: [],
  };
  if (!result.meanings.length) return null;
  return result;
}

function parseCSVRows(raw) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < raw.length; index += 1) {
    const char = raw[index];
    if (quoted) {
      if (char === '"' && raw[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && raw[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  row.push(field);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

function parseECDICT(raw) {
  const rows = parseCSVRows(raw.replace(/^\uFEFF/, ""));
  if (!rows.length) return new Map();
  const header = rows[0].map((value) => value.trim().toLowerCase());
  const wordIndex = Math.max(0, header.indexOf("word"));
  const translationIndex = header.indexOf("translation");
  const definitionIndex = header.indexOf("definition");
  const posIndex = header.indexOf("pos");
  const entries = new Map();
  for (const row of rows.slice(1)) {
    const term = normalizeTerm(row[wordIndex] || "").toLowerCase();
    if (!term) continue;
    const translation = cleanDictionaryField(row[translationIndex] || "");
    const definition = cleanDictionaryField(row[definitionIndex] || "");
    const pos = cleanDictionaryField(row[posIndex] || "");
    const source = translation || definition;
    if (!source) continue;
    const meanings = unique(source.split(/[;；]/).map((value) => value.trim())).map((meaning) => {
      if (!pos || /^[a-z]+\./i.test(meaning)) return meaning;
      return `${pos}. ${meaning}`;
    });
    if (meanings.length) entries.set(term, meanings);
  }
  return entries;
}

function parseOfflineDictionary(raw, path) {
  const trimmed = raw.trim();
  if (!trimmed) return new Map();
  if (/^word\s*,\s*phonetic\s*,\s*definition\s*,\s*translation/i.test(trimmed)) {
    const entries = parseECDICT(trimmed);
    if (!entries.size) throw new Error(`无法解析 ECDICT 文件：${path}`);
    return entries;
  }
  const entries = new Map();
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    const list = Array.isArray(parsed)
      ? parsed
      : Object.entries(parsed).map(([term, meanings]) => ({ term, meanings }));
    for (const entry of list) {
      const term = normalizeTerm(String(entry.term || entry.word || "")).toLowerCase();
      const meanings = Array.isArray(entry.meanings)
        ? entry.meanings
        : [entry.meaning || entry.translation || entry.definition || ""];
      if (term && meanings.some(Boolean)) entries.set(term, unique(meanings.map(String)));
    }
    return entries;
  }
  for (const line of trimmed.split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const columns = line.split(/\t|\s*=>\s*|\s*\|\s*/);
    if (columns.length < 2) continue;
    const term = normalizeTerm(columns.shift()).toLowerCase();
    const meanings = unique(columns.join(" | ").split(/[;；]/).map((part) => part.trim()));
    if (term && meanings.length) entries.set(term, meanings);
  }
  if (!entries.size) throw new Error(`无法解析离线词典：${path}`);
  return entries;
}

function contextFromEditor(editor) {
  const cursor = editor.getCursor();
  const line = editor.getLine(cursor.line) || "";
  if (line.length <= 800) return line;
  return line.slice(Math.max(0, cursor.ch - 400), cursor.ch + 400);
}

function normalizeAIMarkdown(value) {
  const unwrapFormula = (formula, escaped) => {
    const trimmed = formula.trim();
    return escaped ? trimmed.replace(/\\\\/g, "\\") : trimmed;
  };
  return String(value || "")
    .replace(/(\\+)\[([\s\S]*?)(\\+)\]/g, (_match, open, formula, close) => {
      return `$$\n${unwrapFormula(formula, open.length > 1 || close.length > 1)}\n$$`;
    })
    .replace(/(\\+)\(([\s\S]*?)(\\+)\)/g, (_match, open, formula, close) => {
      return `$${unwrapFormula(formula, open.length > 1 || close.length > 1)}$`;
    });
}

function parseAIResultSections(content) {
  const sections = [];
  let current = null;

  const pushCurrent = () => {
    if (!current) return;
    while (current.lines.length && !current.lines[0].trim()) current.lines.shift();
    while (current.lines.length && !current.lines[current.lines.length - 1].trim()) current.lines.pop();
    const value = current.lines.join("\n");
    if (value) sections.push({ label: current.label, value });
    current = null;
  };

  for (const line of String(content || "").replace(/\r\n?/g, "\n").split("\n")) {
    const match = line.match(/^\s*(?:[-*+]\s+)?(?:\d+[.)、]\s*)?([^：:\n]{1,20})[：:]\s*(.*)$/);
    const label = match
      ? match[1].replace(/^(?:\*\*|__)|(?:\*\*|__)$/g, "").trim()
      : "";
    if (match && label && !/[`$\\]/.test(label)) {
      pushCurrent();
      current = { label, lines: [match[2]] };
    } else {
      if (!current) current = { label: "", lines: [] };
      current.lines.push(line);
    }
  }
  pushCurrent();
  return sections;
}

class DefinitionModal extends Modal {
  constructor(app, result) {
    super(app);
    this.result = result;
  }

  onOpen() {
    const { contentEl } = this;
    this.modalEl.addClass("academic-term-lookup-shell");
    contentEl.empty();
    contentEl.addClass("academic-term-lookup-modal");
    const result = this.result;
    const header = contentEl.createDiv({ cls: "academic-term-lookup-header" });
    header.createEl("h2", { text: result.term });
    const metadata = header.createDiv({ cls: "academic-term-lookup-metadata" });
    if (result.source) metadata.createSpan({ text: result.source, cls: "academic-term-lookup-source" });
    if (result.pronunciation) metadata.createSpan({ text: result.pronunciation, cls: "academic-term-lookup-pronunciation" });

    const resultEl = contentEl.createDiv({ cls: "academic-term-lookup-result" });

    if (result.kind === "ai") {
      const article = resultEl.createDiv({ cls: "academic-term-lookup-ai-result" });
      this.renderAIResult(article, result.content);
    } else {
      const list = resultEl.createEl("ol", { cls: "academic-term-lookup-meanings" });
      for (const meaning of result.meanings || []) {
        const item = list.createEl("li");
        const match = meaning.match(/^([a-z]+\.)\s*(.+)$/i);
        if (match) {
          item.createSpan({ text: match[1], cls: "academic-term-lookup-pos" });
          item.createSpan({ text: match[2], cls: "academic-term-lookup-definition" });
        } else {
          item.createSpan({ text: meaning, cls: "academic-term-lookup-definition" });
        }
      }
      if (result.examples?.length) {
        resultEl.createEl("h3", { text: "例句" });
        for (const example of result.examples) resultEl.createEl("p", { text: example, cls: "academic-term-lookup-example" });
      }
    }

    if (result.note) resultEl.createEl("p", { text: result.note, cls: "academic-term-lookup-note" });
    const actions = contentEl.createDiv({ cls: "academic-term-lookup-actions" });
    const closeButton = actions.createEl("button", { text: "关闭" });
    closeButton.addEventListener("click", () => this.close());
  }

  renderAIResult(container, content) {
    for (const section of parseAIResultSections(content)) {
      if (section.label) {
        const row = container.createDiv({ cls: "academic-term-lookup-ai-row" });
        row.createDiv({ text: section.label, cls: "academic-term-lookup-ai-label" });
        this.renderMarkdown(row.createDiv({ cls: "academic-term-lookup-ai-value" }), section.value);
      } else {
        this.renderMarkdown(container.createDiv({ cls: "academic-term-lookup-ai-value" }), section.value);
      }
    }
  }

  renderMarkdown(container, content) {
    const markdown = normalizeAIMarkdown(content);
    Promise.resolve(MarkdownRenderer.render(this.app, markdown, container, "", this)).catch(() => {
      container.empty();
      container.setText(markdown);
    });
  }

  onClose() {
    this.contentEl.empty();
  }
}

class ImportDictionaryModal extends Modal {
  constructor(app, plugin) {
    super(app);
    this.plugin = plugin;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h2", { text: "导入离线词典" });
    contentEl.createEl("p", {
      text: "粘贴 JSON 或 TSV。JSON 示例：{\"policy\":[\"n. 政策；方针\"]}；TSV 每行一个词：policy<Tab>n. 政策；方针",
      cls: "academic-term-lookup-note",
    });
    const textarea = contentEl.createEl("textarea", { cls: "academic-term-lookup-import" });
    textarea.rows = 12;
    textarea.placeholder = '{"policy":["n. 政策；方针"]}';
    const actions = contentEl.createDiv({ cls: "academic-term-lookup-actions" });
    const importButton = actions.createEl("button", { text: "导入" });
    const cancelButton = actions.createEl("button", { text: "取消" });
    importButton.addEventListener("click", async () => {
      try {
        const imported = parseOfflineDictionary(textarea.value, "粘贴内容");
        await this.plugin.saveImportedDictionary(imported);
        new Notice(`已导入 ${imported.size} 个词条`);
        this.close();
      } catch (error) {
        new Notice(`离线词典导入失败：${error.message}`);
      }
    });
    cancelButton.addEventListener("click", () => this.close());
  }

  onClose() {
    this.contentEl.empty();
  }
}

class AcademicTermLookupSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Academic Term Lookup" });

    new Setting(containerEl)
      .setName("查询方式")
      .setDesc("内置词典无需密钥；AI 模式更适合根据上下文判断专业义项。")
      .addDropdown((dropdown) =>
        dropdown
          .addOption("dictionary", "内置双语词典（无需密钥）")
          .addOption("ai", "AI 专业翻译（OpenAI 兼容接口）")
          .setValue(this.plugin.settings.provider)
          .onChange(async (value) => {
            this.plugin.settings.provider = value;
            this.plugin.cache.clear();
            await this.plugin.saveSettings();
            this.display();
          }),
      );

    if (this.plugin.settings.provider === "dictionary") {
      new Setting(containerEl)
        .setName("离线词典优先")
        .setDesc("命中本地词典时不访问网络。")
        .addToggle((toggle) =>
          toggle.setValue(this.plugin.settings.preferOffline).onChange(async (value) => {
            this.plugin.settings.preferOffline = value;
            this.plugin.cache.clear();
            await this.plugin.saveSettings();
          }),
        );
    }

    new Setting(containerEl)
      .setName("仅使用离线词典")
      .setDesc("开启后未命中本地词条时直接提示，不会访问任何在线接口。")
      .addToggle((toggle) =>
        toggle.setValue(this.plugin.settings.offlineOnly).onChange(async (value) => {
          this.plugin.settings.offlineOnly = value;
          this.plugin.cache.clear();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("离线词典文件路径")
      .setDesc("相对于 vault 根目录的 JSON/TSV/CSV 文件路径；留空则只使用内置词条和已导入词条。")
      .addText((text) =>
        text.setPlaceholder("resources/academic-dictionary.json").setValue(this.plugin.settings.offlineDictionaryPath).onChange(async (value) => {
          this.plugin.settings.offlineDictionaryPath = value.trim();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("下载完整英汉词典")
      .setDesc("下载开源 ECDICT（约几十 MB、数十万词条），保存到 vault 后可离线查询。")
      .addButton((button) => button.setButtonText("下载并安装").onClick(async () => {
        button.setDisabled(true).setButtonText("下载中…");
        try {
          const result = await this.plugin.downloadOfflineDictionary();
          this.display();
          new Notice(`ECDICT 已安装：${result.count} 个词条`);
        } catch (error) {
          new Notice(`ECDICT 下载失败：${error.message}`, 10000);
        } finally {
          button.setDisabled(false).setButtonText("下载并安装");
        }
      }));

    new Setting(containerEl)
      .setName("加载离线词典文件")
      .setDesc(`当前已加载 ${this.plugin.offlineDictionary.size} 个词条。`)
      .addButton((button) => button.setButtonText("重新加载").onClick(async () => {
        await this.plugin.loadOfflineDictionary();
        this.display();
        new Notice(`已加载 ${this.plugin.offlineDictionary.size} 个离线词条`);
      }))
      .addButton((button) => button.setButtonText("粘贴导入").onClick(() => new ImportDictionaryModal(this.app, this.plugin).open()));

    if (this.plugin.settings.provider === "ai") {
      new Setting(containerEl)
        .setName("AI 接口格式")
        .setDesc("根据服务商说明选择；Responses 是 OpenAI 新响应格式。")
        .addDropdown((dropdown) => dropdown
          .addOption("chat", "Chat Completions")
          .addOption("responses", "Responses API")
          .setValue(this.plugin.settings.aiFormat)
          .onChange(async (value) => {
            this.plugin.settings.aiFormat = value;
            this.plugin.cache.clear();
            await this.plugin.saveSettings();
          }));

      new Setting(containerEl)
        .setName("AI API 地址")
        .setDesc("可填写基础地址或完整端点；选中词和上下文会发送到该服务。")
        .addText((text) => text.setValue(this.plugin.settings.aiEndpoint).onChange(async (value) => {
          this.plugin.settings.aiEndpoint = value.trim();
          await this.plugin.saveSettings();
        }));

      new Setting(containerEl)
        .setName("模型")
        .addText((text) => text.setPlaceholder("gpt-4o-mini").setValue(this.plugin.settings.aiModel).onChange(async (value) => {
          this.plugin.settings.aiModel = value.trim();
          await this.plugin.saveSettings();
        }));

      new Setting(containerEl)
        .setName("API 密钥")
        .setDesc("密钥只保存在当前 vault 的插件设置中；粘贴时会自动移除空格和换行。")
        .addText((text) => {
          text.inputEl.type = "password";
          text.setValue(this.plugin.settings.aiApiKey).onChange(async (value) => {
            this.plugin.settings.aiApiKey = value.replace(/\s/g, "");
            await this.plugin.saveSettings();
          });
        });

      new Setting(containerEl)
        .setName("测试 AI 连接")
        .setDesc("发送一个不含笔记内容的最小请求，同时验证密钥、接口和模型。")
        .addButton((button) => button.setButtonText("测试连接").onClick(async () => {
          button.setDisabled(true).setButtonText("测试中…");
          try {
            await this.plugin.lookupWithAI("test", "");
            new Notice("AI API 连接成功");
          } catch (error) {
            new Notice(this.plugin.formatLookupError(error), 10000);
          } finally {
            button.setDisabled(false).setButtonText("测试连接");
          }
        }));
    }

    new Setting(containerEl)
      .setName("发送上下文")
      .setDesc("AI 模式会把选中词所在行作为上下文，有助于区分专业义项。")
      .addToggle((toggle) => toggle.setValue(this.plugin.settings.includeContext).onChange(async (value) => {
        this.plugin.settings.includeContext = value;
        await this.plugin.saveSettings();
      }));
  }
}

module.exports = class AcademicTermLookupPlugin extends Plugin {
  async onload() {
    await this.loadSettings();
    this.offlineDictionary = new Map();
    this.cache = new Map();
    this.pending = new Set();
    this.courseRecords = new Map();
    this.vaultIsCourseRoot = Boolean(
      this.app.vault.getAbstractFileByPath("notes")
      || this.app.vault.getAbstractFileByPath(AI_TRANSLATION_FOLDER),
    );
    await this.loadOfflineDictionary();

    this.registerMarkdownPostProcessor((element, context) => this.decorateReadingView(element, context.sourcePath));
    this.registerDomEvent(document, "click", (event) => this.handleTermLinkClick(event), { capture: true });

    this.registerEvent(this.app.workspace.on("editor-menu", (menu, editor) => {
      const selected = normalizeTerm(editor.getSelection() || "");
      if (!selected || !looksLikeEnglish(selected)) return;
      const from = editor.getCursor("from");
      const to = editor.getCursor("to");
      const notePath = this.app.workspace.getActiveFile()?.path;
      menu.addItem((item) => item
        .setTitle(`查询“${selected}”的中文释义`)
        .setIcon("languages")
        .onClick(() => this.lookup(selected, contextFromEditor(editor), notePath, {
          linkSelection: () => this.linkEditorSelection(editor, selected, from, to),
        })));
    }));

    this.registerDomEvent(document, "contextmenu", (event) => {
      const target = event.target instanceof HTMLElement ? event.target.closest(".markdown-preview-view") : null;
      if (!target) return;
      const selected = normalizeTerm(window.getSelection()?.toString() || "");
      if (!selected || !looksLikeEnglish(selected)) return;
      event.preventDefault();
      const menu = new Menu();
      menu.addItem((item) => item
        .setTitle(`查询“${selected}”的中文释义`)
        .setIcon("languages")
        .onClick(() => this.lookup(selected, "", this.app.workspace.getActiveFile()?.path)));
      menu.showAtMouseEvent(event);
    });

    this.addCommand({
      id: "lookup-selected-term",
      name: "查询选中文本的中文释义",
      editorCallback: (editor) => {
        const selected = normalizeTerm(editor.getSelection() || "");
        if (!selected) {
          new Notice("请先选中英文单词或短语");
          return;
        }
        if (!looksLikeEnglish(selected)) {
          new Notice("选中的内容看起来不是英文单词或短语");
          return;
        }
        const from = editor.getCursor("from");
        const to = editor.getCursor("to");
        const notePath = this.app.workspace.getActiveFile()?.path;
        this.lookup(selected, contextFromEditor(editor), notePath, {
          linkSelection: () => this.linkEditorSelection(editor, selected, from, to),
        });
      },
    });

    this.addCommand({
      id: "import-offline-dictionary",
      name: "导入离线词典",
      callback: () => new ImportDictionaryModal(this.app, this).open(),
    });

    this.addCommand({
      id: "download-offline-dictionary",
      name: "下载并安装 ECDICT 英汉词典",
      callback: () => this.downloadOfflineDictionary().then((result) => {
        new Notice(`ECDICT 已安装：${result.count} 个词条`);
      }).catch((error) => {
        new Notice(`ECDICT 下载失败：${error.message}`, 10000);
      }),
    });

    this.addSettingTab(new AcademicTermLookupSettingTab(this.app, this));
  }

  async loadCourseRecords(notePath) {
    const loaded = this.getStorageScope(notePath);
    if (!loaded) return null;
    const cacheKey = loaded.filePath;
    if (this.courseRecords.has(cacheKey)) return this.courseRecords.get(cacheKey);

    let records = normalizeAIRecords();
    try {
      const raw = loaded.absolute
        ? await fs.promises.readFile(loaded.filePath, "utf8")
        : await this.app.vault.adapter.read(loaded.filePath);
      records = normalizeAIRecords(JSON.parse(raw));
    } catch (error) {
      if (!/not found|enoent|no such file/i.test(String(error?.message || error))) {
        console.error("Academic Term Lookup: failed to read AI translations", loaded.filePath, error);
      }
    }
    const result = { ...loaded, records };
    this.courseRecords.set(cacheKey, result);
    return result;
  }

  async saveCourseRecords(loaded) {
    if (loaded.absolute) {
      await fs.promises.mkdir(loaded.folderPath, { recursive: true });
      await fs.promises.writeFile(loaded.filePath, `${JSON.stringify(loaded.records, null, 2)}\n`, "utf8");
    } else {
      try {
        await this.app.vault.adapter.mkdir(loaded.folderPath);
      } catch (error) {
        // The course cache folder may already exist.
      }
      await this.app.vault.adapter.write(loaded.filePath, `${JSON.stringify(loaded.records, null, 2)}\n`);
    }
    this.courseRecords.set(loaded.filePath, loaded);
  }

  getStorageScope(notePath) {
    const physicalScope = resolveCourseStorageScope(this.app.vault.adapter.basePath, notePath);
    if (physicalScope) return physicalScope;

    const courseRoot = courseRootForPath(notePath, this.vaultIsCourseRoot);
    if (courseRoot === null) return null;
    const relativePath = courseRelativePath(notePath, courseRoot);
    if (!relativePath) return null;
    return {
      courseRoot,
      notePath: relativePath,
      filePath: courseTranslationPath(courseRoot),
      folderPath: [courseRoot, AI_TRANSLATION_FOLDER].filter(Boolean).join("/"),
      absolute: false,
    };
  }

  async persistAITranslation(notePath, term, result) {
    if (result?.kind !== "ai") return false;
    const loaded = await this.loadCourseRecords(notePath);
    if (!loaded) return false;
    loaded.records = recordAITranslation(loaded.records, loaded.notePath, term, result);
    await this.saveCourseRecords(loaded);
    return true;
  }

  async getSavedAITranslation(notePath, term) {
    const loaded = await this.loadCourseRecords(notePath);
    if (!loaded) return null;
    const entry = loaded.records.notes[loaded.notePath]?.[normalizeTerm(term).toLowerCase()];
    return entry?.result?.kind === "ai" ? entry : null;
  }

  async openSavedTranslation(notePath, term) {
    const entry = await this.getSavedAITranslation(notePath, term);
    if (!entry) {
      new Notice("这个术语没有可用的 AI 翻译记录，请重新查询。");
      return;
    }
    new DefinitionModal(this.app, entry.result).open();
  }

  handleTermLinkClick(event) {
    const target = event.target instanceof HTMLElement ? event.target.closest("a") : null;
    let term = parseTermLink(target?.getAttribute("href"));
    let notePath = target?.dataset.notePath;

    if (!term) {
      const underline = event.target instanceof HTMLElement
        ? event.target.closest(".cm-underline")
        : null;
      const editor = this.app.workspace.activeEditor?.editor;
      if (!underline || !editor?.containerEl?.contains(underline)) return;
      const rect = underline.getBoundingClientRect();
      let position;
      try {
        position = editor.posAtCoords(rect.left + rect.width / 2, rect.top + rect.height / 2);
      } catch (error) {
        return;
      }
      const cachedLink = position && cachedLinkAtPosition(editor.getLine(position.line), position.ch);
      if (!cachedLink || normalizeTerm(underline.textContent || "").toLowerCase() !== normalizeTerm(cachedLink.label).toLowerCase()) return;
      term = cachedLink.term;
      notePath = this.app.workspace.getActiveFile()?.path;
    }

    event.preventDefault();
    event.stopPropagation();
    void this.openSavedTranslation(notePath || this.app.workspace.getActiveFile()?.path, term);
  }

  linkEditorSelection(editor, term, from, to) {
    const selected = editor.getRange(from, to);
    if (normalizeTerm(selected).toLowerCase() !== normalizeTerm(term).toLowerCase()) return;
    const line = editor.getLine(from.line) || "";
    const before = line.slice(0, from.ch);
    const after = line.slice(to.ch);
    if (/\[[^\]]*$/.test(before) && /^\]\(academic-term-lookup:/.test(after)) return;
    editor.replaceRange(createTermLink(term, selected), from, to);
  }

  async decorateReadingView(element, notePath) {
    if (!notePath) return;
    const loaded = await this.loadCourseRecords(notePath);
    const noteRecords = loaded?.notePath ? loaded.records.notes[loaded.notePath] : null;
    if (!noteRecords || typeof noteRecords !== "object") return;

    element.querySelectorAll(`a[href^="${TERM_LINK_PREFIX}"]`).forEach((link) => {
      link.dataset.notePath = notePath;
    });

    const matcher = termMatcher(Object.keys(noteRecords));
    if (!matcher) return;
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    let node;
    while ((node = walker.nextNode())) {
      if (!node.parentElement?.closest("a, code, pre, .math, .math-inline, .math-block, .katex, script, style")) {
        textNodes.push(node);
      }
    }

    for (const textNode of textNodes) {
      matcher.lastIndex = 0;
      if (!matcher.test(textNode.nodeValue || "")) continue;
      matcher.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      let match;
      while ((match = matcher.exec(textNode.nodeValue || ""))) {
        fragment.append(textNode.nodeValue.slice(cursor, match.index));
        const link = document.createElement("a");
        link.className = "academic-term-lookup-cached-term";
        link.href = `${TERM_LINK_PREFIX}${encodeURIComponent(match[0].toLowerCase())}`;
        link.dataset.notePath = notePath;
        link.textContent = match[0];
        fragment.append(link);
        cursor = match.index + match[0].length;
      }
      fragment.append(textNode.nodeValue.slice(cursor));
      textNode.replaceWith(fragment);
    }
  }

  refreshNoteViews(notePath) {
    this.app.workspace.iterateAllLeaves((leaf) => {
      const view = leaf.view;
      if (view?.file?.path !== notePath) return;
      view.previewMode?.rerender?.(true);
    });
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    this.settings.aiApiKey = String(this.settings.aiApiKey || "").replace(/\s/g, "");
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  async saveImportedDictionary(dictionary) {
    this.offlineDictionary = new Map([...this.offlineDictionary, ...dictionary]);
    this.settings.importedDictionary = Object.fromEntries(this.offlineDictionary);
    await this.saveData({ ...this.settings, importedDictionary: this.settings.importedDictionary });
  }

  async downloadOfflineDictionary() {
    const url = this.settings.offlineDictionaryDownloadUrl || DEFAULT_SETTINGS.offlineDictionaryDownloadUrl;
    const response = await requestUrl({ url, headers: { Accept: "text/csv" } });
    const raw = response.text || "";
    if (!/^word\s*,\s*phonetic\s*,\s*definition\s*,\s*translation/i.test(raw.trim())) {
      throw new Error("下载内容不是 ECDICT CSV；请检查下载源地址");
    }
    const entries = parseECDICT(raw);
    if (entries.size < 1000) throw new Error("ECDICT 文件内容不完整");
    try {
      await this.app.vault.adapter.mkdir(".obsidian/plugins/academic-term-lookup/dictionaries");
    } catch (error) {
      // The directory may already exist.
    }
    await this.app.vault.adapter.write(ECDICT_PATH, raw);
    this.settings.offlineDictionaryPath = ECDICT_PATH;
    await this.saveSettings();
    this.offlineDictionary = new Map([...this.offlineDictionary, ...entries]);
    this.cache.clear();
    return { count: entries.size, path: ECDICT_PATH };
  }

  async loadOfflineDictionary() {
    this.offlineDictionary = new Map(Object.entries(BUILTIN_DICTIONARY));
    const imported = this.settings?.importedDictionary;
    if (imported && typeof imported === "object") {
      for (const [term, meanings] of Object.entries(imported)) {
        this.offlineDictionary.set(term.toLowerCase(), Array.isArray(meanings) ? meanings : [String(meanings)]);
      }
    }
    const path = this.settings?.offlineDictionaryPath?.trim();
    if (!path) return;
    try {
      const raw = await this.app.vault.adapter.read(path);
      const loaded = parseOfflineDictionary(raw, path);
      for (const [term, meanings] of loaded) this.offlineDictionary.set(term, meanings);
    } catch (error) {
      new Notice(`离线词典未加载：${error.message}`);
    }
  }

  async lookup(rawTerm, context, notePath, options = {}) {
    const term = normalizeTerm(rawTerm);
    const cacheKey = `${this.settings.provider}:${this.settings.offlineOnly}:${this.settings.preferOffline}:${term.toLowerCase()}:${context || ""}`;
    if (this.cache.has(cacheKey)) {
      const result = this.cache.get(cacheKey);
      await this.afterLookup(result, term, notePath, options);
      return;
    }
    if (this.pending.has(cacheKey)) return;
    this.pending.add(cacheKey);
    const loading = new Notice(`正在查询“${term}”…`, 0);
    try {
      let result;
      const offline = this.offlineDictionary.get(term.toLowerCase());
      if (this.settings.offlineOnly) {
        if (!offline) throw new Error("离线词典中没有这个词；请导入更完整的词典或关闭“仅使用离线词典”");
        result = { kind: "dictionary", source: "离线词典", term, meanings: offline };
      } else if (this.settings.provider === "ai") {
        result = await this.lookupWithAI(term, context);
      } else if (this.settings.preferOffline && offline) {
        result = { kind: "dictionary", source: "离线词典", term, meanings: offline };
      } else {
        result = await this.lookupWithDictionary(term);
      }
      this.cache.set(cacheKey, result);
      await this.afterLookup(result, term, notePath, options);
    } catch (error) {
      new Notice(this.formatLookupError(error), 10000);
    } finally {
      loading.hide();
      this.pending.delete(cacheKey);
    }
  }

  async afterLookup(result, term, notePath, options) {
    if (result?.kind === "ai" && notePath && this.getStorageScope(notePath)) {
      try {
        await this.persistAITranslation(notePath, term, result);
        options.linkSelection?.();
        this.refreshNoteViews(notePath);
      } catch (error) {
        console.error("Academic Term Lookup: failed to persist AI translation", error);
        new Notice("AI 翻译已显示，但保存课程记录失败。");
      }
    }
    new DefinitionModal(this.app, result).open();
  }

  async lookupWithDictionary(term) {
    try {
      const response = await requestUrl({
        url: `https://dict.youdao.com/jsonapi?doctype=json&q=${encodeURIComponent(term)}`,
        headers: { Accept: "application/json" },
      });
      const parsed = parseYoudao(response.json, term);
      if (parsed) return parsed;
    } catch (error) {
      // Fall through to the second free provider.
    }

    const response = await requestUrl({
      url: `https://api.mymemory.translated.net/get?q=${encodeURIComponent(term)}&langpair=en|zh-CN`,
      headers: { Accept: "application/json" },
    });
    const translation = response.json?.responseData?.translatedText;
    if (!translation) throw new Error("在线词典没有返回释义");
    return {
      kind: "dictionary",
      source: "在线翻译备用接口",
      term,
      meanings: [translation],
      note: "这是通用翻译结果；专业术语建议在设置中启用 AI 专业翻译，或导入离线学术词典。",
    };
  }

  async lookupWithAI(term, context) {
    const apiKey = String(this.settings.aiApiKey || "").replace(/\s/g, "");
    if (!apiKey) throw new Error("请先在插件设置中填写 AI API 密钥");
    const format = this.settings.aiFormat === "responses" ? "responses" : "chat";
    const endpoint = resolveAIEndpoint(this.settings.aiEndpoint, format);
    if (!endpoint) throw new Error("请先在插件设置中填写 AI API 地址");
    const contextText = this.settings.includeContext && context ? context : "（未提供上下文）";
    const systemPrompt = "你是严谨的学术英语-简体中文术语助手。优先使用计算机科学、机器学习、数学和强化学习领域的规范译法；如果存在多个义项，请结合上下文排序。回答简洁，给出词性、最合适的中文译法、必要的领域备注和一个短例句。使用 Obsidian Markdown；行内数学公式只用 $...$，块级数学公式只用 $$...$$，不要使用 \\(...\\) 或 \\[...\\]。不要编造术语。";
    const userPrompt = `待查术语：${term}\n所在上下文：${contextText}\n请用中文回答，格式为：\n词性：...\n推荐译法：...\n其他可能译法：...\n领域备注：...\n例句：...`;
    const body = format === "responses"
      ? {
        model: this.settings.aiModel || "gpt-4o-mini",
        instructions: systemPrompt,
        input: userPrompt,
        stream: false,
      }
      : {
        model: this.settings.aiModel || "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        stream: false,
      };
    const response = await requestUrl({
      url: endpoint,
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const content = extractAIContent(response.json);
    if (!content) throw new Error("AI 接口没有返回内容");
    return { kind: "ai", source: "AI 学术术语翻译", term, content };
  }

  formatLookupError(error) {
    const message = String(error?.message || error || "未知错误");
    if (/\b401\b/.test(message)) {
      return "AI 鉴权失败（401）：API 密钥无效、已撤销或粘贴不完整。请创建新密钥后重新填写。";
    }
    if (/\b403\b/.test(message)) {
      return "AI 请求被拒绝（403）：请检查项目权限或密钥权限。";
    }
    if (/\b404\b/.test(message) || /model.*not found/i.test(message)) {
      return "AI 模型或接口不存在（404）：请检查 API 地址和模型名称。";
    }
    if (/\b429\b/.test(message)) {
      return "AI 请求受限（429）：请检查 API 额度、计费状态或请求频率。";
    }
    return `查询失败：${message}`;
  }
};

module.exports.__test = {
  courseRootForPath,
  courseTranslationPath,
  courseRelativePath,
  resolveCourseStorageScope,
  cachedLinkAtPosition,
  createTermLink,
  parseTermLink,
  recordAITranslation,
};
