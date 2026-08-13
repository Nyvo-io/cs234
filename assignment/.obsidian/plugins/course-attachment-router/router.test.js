const test = require("node:test");
const assert = require("node:assert/strict");

const { getAttachmentFolder, mergeCommunityPluginIds } = require("./router");

test("routes a nested note to its top-level course attachments folder", () => {
  assert.equal(
    getAttachmentFolder("cs234/notes/lec_notes/lec4_notes.md"),
    "cs234/attachments"
  );
});

test("preserves spaces in a course directory name", () => {
  assert.equal(
    getAttachmentFolder("11-711 Advanced Natural Language Processing/notes/lec1.md"),
    "11-711 Advanced Natural Language Processing/attachments"
  );
});

test("uses the vault attachments folder for a root note", () => {
  assert.equal(getAttachmentFolder("inbox.md"), "attachments");
});

test("merges the parent vault's plugin list without duplicating ids", () => {
  assert.deepEqual(
    mergeCommunityPluginIds(["reading-position-memory"], ["academic-term-lookup", "reading-position-memory", "course-attachment-router"]),
    ["reading-position-memory", "academic-term-lookup", "course-attachment-router"],
  );
});
