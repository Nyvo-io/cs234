const { MarkdownView, Notice, Plugin, TFile } = require("obsidian");

const SAVE_DELAY_MS = 500;
const BIND_DELAY_MS = 30;
const RESTORE_RETRY_DELAYS_MS = [0, 60, 180];

module.exports = class ReadingPositionMemoryPlugin extends Plugin {
  async onload() {
    const data = await this.loadData();
    this.positions = this.readPositions(data);
    this.binding = null;
    this.bindTimer = null;
    this.saveTimer = null;
    this.restoreGeneration = 0;
    this.restoreTimers = new Set();

    this.registerEvent(
      this.app.workspace.on("file-open", () => this.scheduleBindAndRestore())
    );
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => this.scheduleBindAndRestore())
    );
    this.registerEvent(
      this.app.workspace.on("layout-change", () => this.scheduleBindAndRestore())
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => this.handleRename(file, oldPath))
    );
    this.registerEvent(
      this.app.vault.on("delete", (file) => this.handleDelete(file))
    );

    this.addCommand({
      id: "forget-current-note-position",
      name: "忘记当前笔记的阅读位置",
      callback: () => this.forgetCurrentPosition(),
    });

    this.app.workspace.onLayoutReady(() => this.scheduleBindAndRestore());
  }

  onunload() {
    this.detachBinding();
    this.clearBindTimer();
    this.clearRestoreTimers();
    this.flushSave();
  }

  readPositions(data) {
    if (!data || typeof data !== "object") return {};
    if (!data.positions || typeof data.positions !== "object") return {};
    return data.positions;
  }

  scheduleBindAndRestore() {
    this.clearBindTimer();
    this.bindTimer = window.setTimeout(() => {
      this.bindTimer = null;
      this.bindAndRestoreActiveView();
    }, BIND_DELAY_MS);
  }

  clearBindTimer() {
    if (this.bindTimer === null) return;
    window.clearTimeout(this.bindTimer);
    this.bindTimer = null;
  }

  bindAndRestoreActiveView() {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    const leaf = this.app.workspace.activeLeaf;
    const filePath = view?.file?.path;
    const mode = view?.getMode?.();
    const scroller = this.getScroller(view, mode);

    if (!view || !leaf || !filePath || !scroller || !["source", "preview"].includes(mode)) {
      // Side panes such as Outline can become active during navigation. Keep the
      // existing Markdown binding so returning focus does not restore stale state.
      return;
    }

    if (
      this.binding?.leaf === leaf &&
      this.binding?.filePath === filePath &&
      this.binding?.mode === mode &&
      this.binding?.scroller === scroller
    ) {
      return;
    }

    this.detachBinding();
    const binding = {
      leaf,
      view,
      filePath,
      mode,
      scroller,
      onScroll: null,
    };
    this.binding = binding;

    const saved = this.positions[filePath]?.[mode];
    if (this.isValidPosition(saved)) {
      this.restoreBinding(binding, saved);
      return;
    }

    this.attachScrollListener(binding);
    this.capturePosition(binding);
  }

  getScroller(view, mode) {
    if (mode === "source") {
      return view?.sourceMode?.cmEditor?.scrollDOM
        || view?.containerEl?.querySelector?.(".cm-scroller")
        || null;
    }
    if (mode === "preview") {
      const previewContainer = view?.previewMode?.containerEl;
      if (previewContainer?.matches?.(".markdown-preview-view")) return previewContainer;
      return previewContainer?.querySelector?.(".markdown-preview-view")
        || view?.containerEl?.querySelector?.(".markdown-preview-view")
        || null;
    }
    return null;
  }

  isValidPosition(position) {
    return Boolean(
      position
      && Number.isFinite(position.top)
      && position.top >= 0
      && Number.isFinite(position.left ?? 0)
    );
  }

  restoreBinding(binding, position) {
    this.clearRestoreTimers();
    const generation = ++this.restoreGeneration;

    const attemptRestore = (attemptIndex) => {
      if (!this.bindingMatches(binding, generation)) return;

      this.applyPosition(binding, position);
      const verifyTimer = window.setTimeout(() => {
        this.restoreTimers.delete(verifyTimer);
        if (!this.bindingMatches(binding, generation)) return;

        const closeEnough = this.positionRestored(binding, position);
        const lastAttempt = attemptIndex === RESTORE_RETRY_DELAYS_MS.length - 1;
        if (closeEnough || lastAttempt) {
          this.attachScrollListener(binding);
          return;
        }

        this.queueRestoreAttempt(() => attemptRestore(attemptIndex + 1), attemptIndex + 1);
      }, 20);
      this.restoreTimers.add(verifyTimer);
    };

    this.queueRestoreAttempt(() => attemptRestore(0), 0);
  }

  queueRestoreAttempt(callback, attemptIndex) {
    const delay = RESTORE_RETRY_DELAYS_MS[attemptIndex] ?? 0;
    const timer = window.setTimeout(() => {
      this.restoreTimers.delete(timer);
      callback();
    }, delay);
    this.restoreTimers.add(timer);
  }

  clearRestoreTimers() {
    for (const timer of this.restoreTimers || []) window.clearTimeout(timer);
    this.restoreTimers?.clear();
    this.restoreGeneration += 1;
  }

  bindingMatches(binding, generation) {
    return Boolean(
      this.binding === binding
      && this.restoreGeneration === generation
      && binding.view.file?.path === binding.filePath
      && binding.view.getMode?.() === binding.mode
    );
  }

  applyPosition(binding, position) {
    if (binding.mode === "source" && typeof binding.view.editor?.scrollTo === "function") {
      binding.view.editor.scrollTo(position.left ?? 0, position.top);
      this.alignSourceAnchor(binding, position);
      return;
    }
    binding.scroller.scrollLeft = position.left ?? 0;
    binding.scroller.scrollTop = position.top;
  }

  readPosition(binding) {
    if (binding.mode === "source" && typeof binding.view.editor?.getScrollInfo === "function") {
      const info = binding.view.editor.getScrollInfo();
      return {
        top: Number.isFinite(info?.top) ? info.top : binding.scroller.scrollTop,
        left: Number.isFinite(info?.left) ? info.left : binding.scroller.scrollLeft,
        anchor: this.readSourceAnchor(binding),
      };
    }
    return {
      top: binding.scroller.scrollTop,
      left: binding.scroller.scrollLeft,
    };
  }

  readSourceAnchor(binding) {
    const cm = binding.view?.sourceMode?.cmEditor?.cm;
    if (!cm?.state?.doc || typeof cm.posAtCoords !== "function") return null;

    const rect = binding.scroller.getBoundingClientRect();
    const offset = cm.posAtCoords({
      x: Math.min(rect.right - 1, rect.left + 20),
      y: rect.top + 5,
    });
    if (!Number.isFinite(offset)) return null;

    const line = cm.state.doc.lineAt(offset);
    const coords = cm.coordsAtPos(offset);
    if (!coords) return null;

    return {
      line: line.number - 1,
      ch: Math.max(0, offset - line.from),
      viewportOffset: coords.top - rect.top,
    };
  }

  resolveSourceAnchor(binding, anchor) {
    const cm = binding.view?.sourceMode?.cmEditor?.cm;
    if (
      !cm?.state?.doc
      || !anchor
      || !Number.isInteger(anchor.line)
      || anchor.line < 0
      || !Number.isFinite(anchor.ch)
      || !Number.isFinite(anchor.viewportOffset)
    ) {
      return null;
    }

    const lineNumber = Math.min(anchor.line + 1, cm.state.doc.lines);
    const line = cm.state.doc.line(lineNumber);
    return {
      cm,
      offset: Math.min(line.to, line.from + Math.max(0, anchor.ch)),
      editorPosition: {
        line: lineNumber - 1,
        ch: Math.min(line.length, Math.max(0, anchor.ch)),
      },
      viewportOffset: anchor.viewportOffset,
    };
  }

  alignSourceAnchor(binding, position) {
    const resolved = this.resolveSourceAnchor(binding, position.anchor);
    if (!resolved) return;

    const coords = resolved.cm.coordsAtPos(resolved.offset);
    if (!coords) {
      binding.view.editor.scrollIntoView(
        { from: resolved.editorPosition, to: resolved.editorPosition },
        true
      );
      return;
    }

    const rect = binding.scroller.getBoundingClientRect();
    const delta = coords.top - (rect.top + resolved.viewportOffset);
    if (Math.abs(delta) <= 1) return;

    const current = binding.view.editor.getScrollInfo();
    binding.view.editor.scrollTo(current.left ?? 0, current.top + delta);
  }

  positionRestored(binding, position) {
    if (binding.mode === "source" && position.anchor) {
      const resolved = this.resolveSourceAnchor(binding, position.anchor);
      if (!resolved) return false;
      const coords = resolved.cm.coordsAtPos(resolved.offset);
      if (!coords) return false;
      const rect = binding.scroller.getBoundingClientRect();
      return Math.abs(coords.top - (rect.top + resolved.viewportOffset)) <= 2;
    }

    const current = this.readPosition(binding);
    return Math.abs(current.top - position.top) <= 2;
  }

  attachScrollListener(binding) {
    if (this.binding !== binding || binding.onScroll) return;
    binding.onScroll = () => this.capturePosition(binding);
    binding.scroller.addEventListener("scroll", binding.onScroll, { passive: true });
    this.capturePosition(binding);
  }

  detachBinding() {
    this.clearRestoreTimers();
    if (this.binding?.onScroll) {
      this.binding.scroller.removeEventListener("scroll", this.binding.onScroll);
    }
    this.binding = null;
  }

  capturePosition(binding) {
    if (
      this.binding !== binding
      || binding.view.file?.path !== binding.filePath
      || binding.view.getMode?.() !== binding.mode
    ) {
      return;
    }

    const position = this.readPosition(binding);
    if (!Number.isFinite(position.top) || position.top < 0) return;

    this.positions[binding.filePath] ||= {};
    this.positions[binding.filePath][binding.mode] = {
      top: position.top,
      left: Number.isFinite(position.left) ? position.left : 0,
      ...(position.anchor ? { anchor: position.anchor } : {}),
      updatedAt: Date.now(),
    };
    this.scheduleSave();
  }

  scheduleSave() {
    if (this.saveTimer !== null) window.clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => {
      this.saveTimer = null;
      void this.persistPositions();
    }, SAVE_DELAY_MS);
  }

  async persistPositions() {
    await this.saveData({ version: 1, positions: this.positions });
  }

  flushSave() {
    if (this.saveTimer !== null) {
      window.clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    void this.persistPositions();
  }

  handleRename(file, oldPath) {
    if (!(file instanceof TFile) || !this.positions[oldPath]) return;
    this.positions[file.path] = this.positions[oldPath];
    delete this.positions[oldPath];
    this.scheduleSave();
  }

  handleDelete(file) {
    if (!(file instanceof TFile) || !this.positions[file.path]) return;
    delete this.positions[file.path];
    this.scheduleSave();
  }

  forgetCurrentPosition() {
    const filePath = this.app.workspace.getActiveFile()?.path;
    if (!filePath || !this.positions[filePath]) {
      new Notice("当前笔记没有已保存的阅读位置");
      return;
    }
    delete this.positions[filePath];
    this.scheduleSave();
    new Notice("已清除当前笔记的阅读位置");
  }
};
