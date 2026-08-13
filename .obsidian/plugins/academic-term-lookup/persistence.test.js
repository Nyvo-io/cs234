const assert = require("node:assert/strict");
const Module = require("node:module");
const test = require("node:test");

const originalLoad = Module._load;
Module._load = function loadObsidianStub(request, parent, isMain) {
  if (request === "obsidian") {
    return {
      MarkdownRenderer: { render: async () => {} },
      Modal: class {},
      Menu: class {},
      Notice: class {},
      Plugin: class {},
      PluginSettingTab: class {},
      Setting: class {},
      requestUrl: async () => ({}),
    };
  }
  return originalLoad.call(this, request, parent, isMain);
};

const Plugin = require("./main");
Module._load = originalLoad;

const {
  courseRootForPath,
  courseTranslationPath,
  courseRelativePath,
  resolveCourseStorageScope,
  createTermLink,
  cachedLinkAtPosition,
  parseTermLink,
  recordAITranslation,
} = Plugin.__test || {};

test("stores each course's AI translations in that course's hidden data folder", () => {
  assert.equal(courseRootForPath("cs234/notes/lec_notes/lec4_notes.md"), "cs234");
  assert.equal(
    courseRootForPath("11-711 Advanced Natural Language Processing/notes/lec_notes/03-lm.md"),
    "11-711 Advanced Natural Language Processing",
  );
  assert.equal(courseTranslationPath("cs234"), "cs234/.academic-term-lookup/ai-translations.json");
});

test("does not persist a record for a vault-root note", () => {
  assert.equal(courseRootForPath("inbox.md"), null);
});

test("uses the vault root when the opened vault is itself a course", () => {
  assert.equal(courseRootForPath("notes/lec_notes/lec4_notes.md", true), "");
  assert.equal(courseTranslationPath(""), ".academic-term-lookup/ai-translations.json");
  assert.equal(
    courseRelativePath("notes/lec_notes/lec4_notes.md", ""),
    "notes/lec_notes/lec4_notes.md",
  );
});

test("maps the same physical course note to one canonical translation record", () => {
  const notePath = "notes/lec_notes/lec4_notes.md";
  const expected = {
    courseRoot: "cs234",
    notePath,
    filePath: "/Users/nyvo/course/cs234/.academic-term-lookup/ai-translations.json",
    folderPath: "/Users/nyvo/course/cs234/.academic-term-lookup",
    absolute: true,
  };

  assert.deepEqual(
    resolveCourseStorageScope("/Users/nyvo/course", "cs234/" + notePath),
    expected,
  );
  assert.deepEqual(resolveCourseStorageScope("/Users/nyvo/course/cs234", notePath), expected);
  assert.deepEqual(resolveCourseStorageScope("/Users/nyvo/course/cs234/notes", "lec_notes/lec4_notes.md"), expected);
});

test("does not depend on a hard-coded course directory name", () => {
  const scope = resolveCourseStorageScope(
    "/Users/nyvo/course/11-711 Advanced Natural Language Processing",
    "notes/lec1.md",
  );
  assert.equal(scope.courseRoot, "11-711 Advanced Natural Language Processing");
  assert.equal(scope.notePath, "notes/lec1.md");
  assert.equal(
    scope.filePath,
    "/Users/nyvo/course/11-711 Advanced Natural Language Processing/.academic-term-lookup/ai-translations.json",
  );
});

test("creates and parses a compact clickable academic-term link", () => {
  assert.equal(createTermLink("policy iteration", "Policy iteration"), "[Policy iteration](academic-term-lookup:policy%20iteration)");
  assert.equal(createTermLink("backup", " backup"), " [backup](academic-term-lookup:backup)");
  assert.equal(parseTermLink("academic-term-lookup:policy%20iteration"), "policy iteration");
  assert.equal(parseTermLink("https://example.com"), null);
});

test("finds a cached link at a Live Preview source position", () => {
  const line = "课件先假设有 [oracle](academic-term-lookup:oracle) 能返回真实值。";
  const position = line.indexOf("[oracle]") + 2;
  assert.deepEqual(cachedLinkAtPosition(line, position), {
    term: "oracle",
    label: "oracle",
  });
  const paddedLine = "Bellman operator [ backup](academic-term-lookup:backup) 后继续更新。";
  const paddedPosition = paddedLine.indexOf("[ backup]") + 3;
  assert.deepEqual(cachedLinkAtPosition(paddedLine, paddedPosition), {
    term: "backup",
    label: "backup",
  });
  assert.equal(cachedLinkAtPosition(line, line.indexOf("能返回")), null);
});

test("records the AI result under the course-relative note path and normalized term", () => {
  const result = { kind: "ai", term: "oracle", content: "推荐译法：预言机" };
  const records = recordAITranslation({}, "notes/lec_notes/lec4_notes.md", "Oracle", result, "2026-07-29T00:00:00.000Z");

  assert.deepEqual(records, {
    version: 1,
    notes: {
      "notes/lec_notes/lec4_notes.md": {
        oracle: {
          term: "Oracle",
          result,
          updatedAt: "2026-07-29T00:00:00.000Z",
        },
      },
    },
  });
});

test("ignores non-AI results when writing the course record", () => {
  assert.deepEqual(
    recordAITranslation({}, "notes/lec4.md", "oracle", {
      kind: "dictionary",
      term: "oracle",
      meanings: ["预言机"],
    }),
    { version: 1, notes: {} },
  );
});
