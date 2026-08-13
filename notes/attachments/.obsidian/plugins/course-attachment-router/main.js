const fs = require("fs");
const path = require("path");
const { Notice, Plugin, TFile, TFolder } = require("obsidian");

// Obsidian loads local plugin entrypoints as standalone scripts, so this small
// routing rule must live in main.js instead of relying on a relative require.
function getAttachmentFolder(filePath) {
  const [topLevel, ...rest] = String(filePath || "").split("/");
  if (!topLevel || rest.length === 0 || topLevel.startsWith(".")) {
    return "attachments";
  }
  return `${topLevel}/attachments`;
}

function mergeCommunityPluginIds(existing, available) {
  return [...new Set(
    [...(Array.isArray(existing) ? existing : []), ...(Array.isArray(available) ? available : [])]
      .filter((id) => typeof id === "string" && id.trim()),
  )];
}

module.exports = class CourseAttachmentRouterPlugin extends Plugin {
  async onload() {
    this.vaultBasePath = this.app.vault.adapter.basePath;
    const data = await this.loadData();
    this.fallbackFolder = data?.fallbackFolder
      ?? this.app.vault.getConfig("attachmentFolderPath")
      ?? "/";
    this.managedFolder = null;

    if (!data?.fallbackFolder) {
      await this.saveData({ fallbackFolder: this.fallbackFolder });
    }

    await this.provisionNestedVaults();
    this.registerEvent(this.app.vault.on("create", (file) => {
      if (file instanceof TFolder && !file.name.startsWith(".")) {
        void this.provisionNestedVaults();
      }
    }));
    this.registerInterval(window.setInterval(() => {
      void this.provisionNestedVaults();
    }, 15000));

    this.registerEvent(
      this.app.workspace.on("file-open", (file) => this.syncForFile(file))
    );
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        this.syncForFile(this.app.workspace.getActiveFile());
      })
    );
    this.registerEvent(
      this.app.workspace.on("editor-paste", (_event, _editor, view) => {
        this.syncForFile(view?.file ?? this.app.workspace.getActiveFile());
      })
    );

    this.addCommand({
      id: "sync-attachment-folder-for-current-note",
      name: "为当前笔记刷新附件目录",
      callback: () => {
        const folder = this.syncForFile(this.app.workspace.getActiveFile());
        if (folder) new Notice(`附件目录：${folder}`);
      },
    });

    this.app.workspace.onLayoutReady(() => {
      this.syncForFile(this.app.workspace.getActiveFile());
    });
  }

  async readParentPluginIds() {
    const pluginRoot = path.join(this.vaultBasePath, ".obsidian", "plugins");
    let ids = [];
    try {
      const entries = await fs.promises.readdir(pluginRoot, { withFileTypes: true });
      ids = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    } catch (error) {
      return ids;
    }

    try {
      const raw = await fs.promises.readFile(
        path.join(this.vaultBasePath, ".obsidian", "community-plugins.json"),
        "utf8",
      );
      return mergeCommunityPluginIds(JSON.parse(raw), ids);
    } catch (error) {
      return ids;
    }
  }

  async provisionNestedVaults() {
    if (!this.vaultBasePath) return;
    const folders = new Set();
    const root = this.app.vault.getRoot?.();
    for (const child of root?.children || []) {
      if (child instanceof TFolder && !child.name.startsWith(".")) folders.add(child.name);
    }
    await this.collectNestedVaultPaths(this.vaultBasePath, "", folders);
    const pluginIds = await this.readParentPluginIds();
    for (const folder of folders) await this.provisionNestedVault(folder, pluginIds);
  }

  async collectNestedVaultPaths(currentPath, relativePath, output) {
    let entries;
    try {
      entries = await fs.promises.readdir(currentPath, { withFileTypes: true });
    } catch (error) {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith(".") || [
        "node_modules",
        "__pycache__",
        ".venv",
        "dist",
        "build",
      ].includes(entry.name)) continue;

      const childPath = path.join(currentPath, entry.name);
      const childRelative = relativePath ? `${relativePath}/${entry.name}` : entry.name;
      let childEntries;
      try {
        childEntries = await fs.promises.readdir(childPath, { withFileTypes: true });
      } catch (error) {
        continue;
      }
      if (!relativePath || childEntries.some((child) => child.isDirectory() && child.name === ".obsidian")) {
        output.add(childRelative);
      }
      await this.collectNestedVaultPaths(childPath, childRelative, output);
    }
  }

  async provisionNestedVault(folderName, pluginIds = null) {
    if (!this.vaultBasePath || !folderName || folderName.startsWith(".")) return;
    const ids = pluginIds || await this.readParentPluginIds();
    const targetRoot = path.join(this.vaultBasePath, folderName);
    const targetPlugins = path.join(targetRoot, ".obsidian", "plugins");
    await fs.promises.mkdir(targetPlugins, { recursive: true });

    for (const id of ids) {
      const source = path.join(this.vaultBasePath, ".obsidian", "plugins", id);
      const target = path.join(targetPlugins, id);
      let entries;
      try {
        entries = await fs.promises.readdir(source, { withFileTypes: true });
      } catch (error) {
        continue;
      }
      await fs.promises.mkdir(target, { recursive: true });
      for (const entry of entries) {
        if (!entry.isFile()) continue;
        const sourceFile = path.join(source, entry.name);
        const targetFile = path.join(target, entry.name);
        if (entry.name === "data.json") {
          try {
            await fs.promises.access(targetFile);
            continue;
          } catch (error) {
            // New nested vaults inherit the parent plugin settings once.
          }
        }
        await fs.promises.copyFile(sourceFile, targetFile);
      }
    }

    const communityPath = path.join(targetRoot, ".obsidian", "community-plugins.json");
    let existing = [];
    try {
      existing = JSON.parse(await fs.promises.readFile(communityPath, "utf8"));
    } catch (error) {
      // The nested vault may not have Obsidian settings yet.
    }
    const merged = mergeCommunityPluginIds(existing, ids);
    await fs.promises.writeFile(communityPath, `${JSON.stringify(merged, null, 2)}\n`, "utf8");
  }

  onunload() {
    if (
      this.managedFolder
      && this.app.vault.getConfig("attachmentFolderPath") === this.managedFolder
    ) {
      this.app.vault.setConfig("attachmentFolderPath", this.fallbackFolder);
    }
  }

  syncForFile(file) {
    if (!(file instanceof TFile) || file.extension !== "md") return null;

    const folder = getAttachmentFolder(file.path);
    if (this.app.vault.getConfig("attachmentFolderPath") !== folder) {
      this.app.vault.setConfig("attachmentFolderPath", folder);
    }
    this.managedFolder = folder;
    void this.ensureFolder(folder);
    return folder;
  }

  async ensureFolder(folder) {
    const existing = this.app.vault.getAbstractFileByPath(folder);
    if (existing instanceof TFolder) return;
    if (existing) {
      new Notice(`无法创建附件目录：${folder} 已被同名文件占用`);
      return;
    }

    try {
      await this.app.vault.createFolder(folder);
    } catch (error) {
      if (!(this.app.vault.getAbstractFileByPath(folder) instanceof TFolder)) {
        console.error("Course Attachment Router: failed to create folder", folder, error);
      }
    }
  }
};
