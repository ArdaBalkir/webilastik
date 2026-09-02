/// <reference lib="webworker" />

import { inflateRaw } from "./dzip_helper";

const ATLAS_ROOT =
  "https://data-proxy.ebrains.eu/api/v1/buckets/quint-atlas-binaries/LZ-flatpacks/";

interface AtlasDescriptor {
  name: string;
  encoding: number;
  xdim: number;
  ydim: number;
  zdim: number;
  labels: Array<{ rgb?: string; name?: string }>;
  transformations?: unknown[];
  remap?: number[];
}

type WorkerRequest = { atlasName: string };
type WorkerResponse =
  | { type: "progress"; message: string }
  | { type: "ready"; atlas: AtlasDescriptor; buffer: ArrayBuffer }
  | { type: "error"; message: string };

const workerScope = self as DedicatedWorkerGlobalScope;

workerScope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  try {
    const atlasName = event.data.atlasName.trim();
    if (!atlasName) throw new Error("Atlas name is missing");
    const base = ATLAS_ROOT + encodeURIComponent(atlasName);

    progress("Loading atlas descriptor…");
    const descriptorResponse = await fetch(base + ".json");
    if (!descriptorResponse.ok) {
      throw new Error(`Atlas descriptor request failed (${descriptorResponse.status})`);
    }
    const atlas = (await descriptorResponse.json()) as AtlasDescriptor;
    validateDescriptor(atlas);

    progress("Downloading compressed atlas…");
    const packResponse = await fetch(base + ".pack");
    if (!packResponse.ok) {
      throw new Error(`Atlas data request failed (${packResponse.status})`);
    }
    const packed = new Uint8Array(await packResponse.arrayBuffer());

    progress("Inflating atlas…");
    const rle = await inflateRaw(packed);
    progress("Decoding atlas labels…");
    const voxelCount = atlas.xdim * atlas.ydim * atlas.zdim;
    const decoded = decodeRle(rle, atlas.encoding, voxelCount);

    const buffer = decoded.buffer as ArrayBuffer;
    const response: WorkerResponse = {
      type: "ready",
      atlas,
      buffer,
    };
    workerScope.postMessage(response, [buffer]);
  } catch (error) {
    const response: WorkerResponse = {
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    };
    workerScope.postMessage(response);
  }
};

function progress(message: string) {
  const response: WorkerResponse = { type: "progress", message };
  workerScope.postMessage(response);
}

function validateDescriptor(atlas: AtlasDescriptor) {
  if (
    !Number.isInteger(atlas.xdim) || atlas.xdim <= 0 ||
    !Number.isInteger(atlas.ydim) || atlas.ydim <= 0 ||
    !Number.isInteger(atlas.zdim) || atlas.zdim <= 0 ||
    !Array.isArray(atlas.labels)
  ) {
    throw new Error("Atlas descriptor has invalid dimensions or labels");
  }
}

function decodeRle(
  data: Uint8Array,
  encoding: number,
  expectedLength: number,
): Uint8Array | Uint16Array {
  if (encoding !== 1 && encoding !== 2) {
    throw new Error(`Unsupported atlas encoding ${encoding}`);
  }

  const result = encoding === 1
    ? new Uint8Array(expectedLength)
    : new Uint16Array(expectedLength);
  let readPosition = 0;
  let writePosition = 0;

  const read8 = () => {
    if (readPosition >= data.length) throw new Error("Truncated atlas data");
    return data[readPosition++];
  };
  const read15 = () => {
    const first = read8();
    return first > 127 ? ((first - 128) << 8) + read8() : first;
  };

  while (readPosition < data.length) {
    const value = encoding === 1 ? read8() : read15();
    const count = read15() + 1;
    const end = writePosition + count;
    if (end > result.length) {
      throw new Error("Decoded atlas is larger than its declared dimensions");
    }
    result.fill(value, writePosition, end);
    writePosition = end;
  }

  if (writePosition !== expectedLength) {
    throw new Error(
      `Decoded atlas has ${writePosition} voxels; expected ${expectedLength}`,
    );
  }
  return result;
}

export {};
