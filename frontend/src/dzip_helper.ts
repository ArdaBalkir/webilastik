// TypeScript port of netunzip utility for partial remote ZIP central directory reading.
// Minimal typing added; focuses on correctness and parity with previous JS implementation.

export interface ZipDirectoryEntryMeta {
  vermade: number;
  verext: number;
  gpflags: number;
  method: number;
  timestamp: Date;
  crc: number;
  compsize: number;
  uncompsize: number;
  namelength: number;
  extralength: number;
  commentlength: number;
  diskno: number;
  intattrs: number;
  extattrs: number;
  offset: number;
  name?: string;
  extra?: Uint8Array;
  comment?: string;
}

export interface NetUnzipDirectory {
  entries: Map<string, ZipDirectoryEntryMeta>;
  get(
    entry: ZipDirectoryEntryMeta,
    allowCompressed?: boolean,
  ): Promise<Uint8Array>;
}

export type UrlLocator = string | (() => Promise<string> | string);

async function netunzip(
  target: UrlLocator,
  extraHeaders?: Record<string, string>,
): Promise<NetUnzipDirectory> {
  const hasAuth = !!extraHeaders?.["Authorization"];
  const rawUrl = typeof target === "function" ? await target() : target;
  const isDataProxy = hasAuth && rawUrl.includes("data-proxy.ebrains.eu");

  // Pre-signed S3 URLs from data-proxy expire in ~10 seconds, so we must
  // call ?redirect=false fresh for every range request to get a new URL.
  async function rangeGet(range: string): Promise<ArrayBuffer> {
    if (isDataProxy) {
      const proxyUrl = new URL(rawUrl);
      proxyUrl.searchParams.set("redirect", "false");
      const res = await fetch(proxyUrl.toString(), { headers: extraHeaders });
      if (!res.ok)
        throw new Error(`data-proxy error ${res.status}: ${await res.text()}`);
      const json = await res.json();
      const preSignedUrl: string =
        json.url ?? json.URL ?? String(Object.values(json)[0]);
      return fetch(preSignedUrl, { headers: { range } }).then((r) =>
        r.arrayBuffer(),
      );
    }
    return fetch(rawUrl, { headers: { range, ...extraHeaders } }).then((r) =>
      r.arrayBuffer(),
    );
  }

  const footerView = await rangeGet("bytes=-22").then((b) => new DataView(b));
  if (footerView.getUint32(0, true) !== 0x06054b50) {
    throw new Error(
      "EODC header signature not found. Not a ZIP or has comment field.",
    );
  }
  const directorySize = footerView.getUint32(12, true);
  let directoryOffset = footerView.getUint32(16, true);
  if (directoryOffset === 0xffffffff) {
    const locatorView = await rangeGet("bytes=-42").then(
      (b) => new DataView(b),
    );
    if (locatorView.getUint32(0, true) !== 0x07064b50) {
      throw new Error("EODC64 locator signature not found.");
    }
    const zip64Offset = Number(locatorView.getBigUint64(8, true));
    const zip64View = await rangeGet(`bytes=${zip64Offset}-`).then(
      (b) => new DataView(b),
    );
    if (zip64View.getUint32(0, true) !== 0x06064b50) {
      throw new Error("EODC64 header signature not found.");
    }
    directoryOffset = Number(zip64View.getBigUint64(48, true));
  }
  const centralBytes = await rangeGet(
    `bytes=${directoryOffset}-${directoryOffset + directorySize - 1}`,
  );
  const decoder = new TextDecoder();
  const entries = new Map<string, ZipDirectoryEntryMeta>();
  let cursor = 0;
  while (cursor < centralBytes.byteLength) {
    const view = new DataView(centralBytes, cursor);
    cursor += 46;
    if (view.getUint32(0, true) !== 0x02014b50)
      throw new Error("Central directory entry signature missing.");
    const e = view.getUint16(12, true);
    const hour = e >> 11;
    const minute = (e >> 5) & 0x3f;
    const second = 2 * (e & 0x1f);
    const d = view.getUint16(14, true);
    const year = 1980 + (d >> 9);
    const month = (d >> 5) & 0x0f;
    const day = d & 0x1f;
    const meta: ZipDirectoryEntryMeta = {
      vermade: view.getUint16(4, true),
      verext: view.getUint16(6, true),
      gpflags: view.getUint16(8, true),
      method: view.getUint16(10, true),
      timestamp: new Date(year, month, day, hour, minute, second),
      crc: view.getUint32(16, true),
      compsize: view.getUint32(20, true),
      uncompsize: view.getUint32(24, true),
      namelength: view.getUint16(28, true),
      extralength: view.getUint16(30, true),
      commentlength: view.getUint16(32, true),
      diskno: view.getUint16(34, true),
      intattrs: view.getUint16(36, true),
      extattrs: view.getUint32(38, true),
      offset: view.getUint32(42, true),
    };
    meta.name = decoder.decode(
      new Uint8Array(centralBytes, cursor, meta.namelength),
    );
    cursor += meta.namelength;
    meta.extra = new Uint8Array(centralBytes, cursor, meta.extralength);
    if (meta.offset === 0xffffffff) {
      const extraView = new DataView(centralBytes, cursor, meta.extralength);
      let p = 0;
      while (p < extraView.byteLength) {
        const headerId = extraView.getUint16(p, true);
        const dataSize = extraView.getUint16(p + 2, true);
        if (headerId === 0x0001) {
          // Zip64 extended info
          if (dataSize !== 8)
            throw new Error("Unsupported Zip64 extra field length.");
          meta.offset = Number(extraView.getBigUint64(p + 4, true));
          break;
        }
        p += 4 + dataSize;
      }
    }
    cursor += meta.extralength;
    meta.comment = decoder.decode(
      new Uint8Array(centralBytes, cursor, meta.commentlength),
    );
    cursor += meta.commentlength;
    entries.set(meta.name!, meta);
  }
  return {
    entries,
    async get(
      entry: ZipDirectoryEntryMeta,
      allowCompressed?: boolean,
    ): Promise<Uint8Array> {
      const method = entry.method;
      if (!allowCompressed && method !== 0 && method !== 8)
        throw new Error(`Unsupported compression method ${method}`);
      const localHeader = await rangeGet(
        `bytes=${entry.offset}-${entry.offset + 30 - 1}`,
      ).then((b) => new DataView(b));
      if (localHeader.getUint32(0, true) !== 0x04034b50)
        throw new Error("Local file header signature missing.");
      const compSize = localHeader.getUint32(18, true);
      const nameLen = localHeader.getUint16(26, true);
      const extraLen = localHeader.getUint16(28, true);
      const dataOffset = entry.offset + 30 + nameLen + extraLen;
      const rawBuf = await rangeGet(
        `bytes=${dataOffset}-${dataOffset + compSize - 1}`,
      );
      const data = new Uint8Array(rawBuf);
      if (allowCompressed || method === 0) return data; // stored
      if (method === 8) return inflateRaw(data);
      throw new Error(`Unsupported method ${method}`);
    },
  };
}

// Uses the browser's native DecompressionStream for raw deflate (RFC 1951).
// Replaces the prior incomplete manual inflate implementation.
export async function inflateRaw(src: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate-raw");
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();
  writer.write(src.buffer as ArrayBuffer);
  writer.close();
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value!);
  }
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

export default netunzip;
