// Browser-side checks on an uploaded VCF, run before anything is sent.
// Pure functions of the file's name, size and text: no DOM, no network.
// They are a convenience for the visitor, not a security boundary: the
// inference API re-validates everything (docs/llds/07_ui.md § Input Validation).

// 4 MiB. The VCF travels JSON-encoded, where every tab becomes the two-byte
// escape \t, and Lambda rejects synchronous request bodies over 6 MB; a 4 MiB
// file stays under that after encoding.
export const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

// @spec UI-UI-003
export function checkExtension(name) {
  return name.toLowerCase().endsWith(".vcf")
    ? null
    : "The file must be a .vcf file.";
}

// @spec UI-UI-004
export function checkSize(sizeBytes) {
  return sizeBytes > MAX_UPLOAD_BYTES
    ? "The file is larger than 4 MiB."
    : null;
}

/**
 * Number of sample columns declared on the #CHROM header line.
 * @param {string} text VCF contents.
 * @returns {number|null} Columns after FORMAT (0 if there is no FORMAT column),
 *   or null if the file has no #CHROM line.
 */
export function sampleCount(text) {
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("##")) continue;
    if (line.startsWith("#CHROM")) {
      const cols = line.split("\t");
      return cols.length > 9 ? cols.length - 9 : 0;
    }
    if (line.trim() !== "") return null; // data before any header
  }
  return null;
}

// @spec UI-UI-005
export function checkSingleSample(text) {
  const n = sampleCount(text);
  if (n === null) return "The file has no #CHROM header line.";
  if (n !== 1) return `The file must contain exactly one sample (found ${n}).`;
  return null;
}

/**
 * Checks that need only the file's metadata, so an oversized file is
 * rejected without being read.
 * @returns {string[]} Error messages; empty if the checks pass.
 */
export function checkFileMeta(name, sizeBytes) {
  return [checkExtension(name), checkSize(sizeBytes)].filter(Boolean);
}

/** @returns {string[]} Error messages from the content checks; empty if they pass. */
export function checkFileText(text) {
  return [checkSingleSample(text)].filter(Boolean);
}
