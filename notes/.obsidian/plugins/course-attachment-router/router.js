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

module.exports = { getAttachmentFolder, mergeCommunityPluginIds };
