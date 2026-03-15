import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { dataProxy } from "../api";
import type { BucketListEntry, StorageObject, StorageDir } from "../types";

const DATA_PROXY_BASE = "https://data-proxy.ebrains.eu/api/v1";

interface Props {
  token: string;
  onSelect: (url: string) => void;
  onClose: () => void;
}

export function DataProxyDialog({ token, onSelect, onClose }: Props) {
  const buckets = useSignal<BucketListEntry[]>([]);
  const objects = useSignal<(StorageObject | StorageDir)[]>([]);
  const bucket = useSignal("");
  const prefix = useSignal("");
  const destName = useSignal("predictions.dzip");
  const msg = useSignal("Loading buckets…");

  useEffect(() => {
    dataProxy
      .listBuckets(token)
      .then((b) => {
        buckets.value = b;
        msg.value = "";
      })
      .catch((e) => {
        msg.value = `Could not load buckets: ${e}`;
      });
  }, []);

  function fetchObjects(bkt: string, pfx: string) {
    msg.value = "";
    dataProxy
      .listObjects(token, bkt, pfx)
      .then((r) => {
        objects.value = r.objects;
      })
      .catch((e) => {
        msg.value = String(e);
      });
  }

  function changeBucket(name: string) {
    bucket.value = name;
    prefix.value = "";
    objects.value = [];
    if (name) fetchObjects(name, "");
  }

  function pickItem(item: StorageObject | StorageDir) {
    if ("subdir" in item) {
      prefix.value = item.subdir;
      fetchObjects(bucket.value, item.subdir);
    } else {
      destName.value = item.name.split("/").pop() ?? item.name;
    }
  }

  function goUp() {
    const parts = prefix.value.replace(/\/$/, "").split("/");
    parts.pop();
    const newPfx = parts.length ? parts.join("/") + "/" : "";
    prefix.value = newPfx;
    fetchObjects(bucket.value, newPfx);
  }

  function confirm() {
    if (!bucket.value || !destName.value) return;
    const objName = (prefix.value + destName.value).replace(/^\//, "");
    onSelect(`${DATA_PROXY_BASE}/buckets/${bucket.value}/${objName}`);
    onClose();
  }

  return (
    <div class="dialog-overlay" onClick={onClose}>
      <div class="dialog" onClick={(e: Event) => e.stopPropagation()}>
        <div class="dialog-head">
          <b>EBRAINS destination</b>
          <button class="btn-icon" onClick={onClose}>
            ×
          </button>
        </div>

        {msg.value && <p class="status">{msg.value}</p>}

        <select
          class="input-url"
          value={bucket.value}
          onChange={(e: Event) =>
            changeBucket((e.target as HTMLSelectElement).value)
          }
        >
          <option value="">— select bucket —</option>
          {buckets.value.map((b) => (
            <option key={b.name} value={b.name}>
              {b.name}
            </option>
          ))}
        </select>

        {(objects.value.length > 0 || prefix.value) && (
          <ul class="bucket-list">
            {prefix.value && (
              <li class="bucket-item" onClick={goUp}>
                ⬆ ..
              </li>
            )}
            {objects.value.map((o) =>
              "subdir" in o ? (
                <li
                  key={o.subdir}
                  class="bucket-item"
                  onClick={() => pickItem(o)}
                >
                  📁 {o.subdir.slice(prefix.value.length)}
                </li>
              ) : (
                <li
                  key={o.name}
                  class="bucket-item"
                  onClick={() => pickItem(o)}
                >
                  📄 {o.name.slice(prefix.value.length)}
                </li>
              ),
            )}
          </ul>
        )}

        <div class="row">
          <span class="bucket-prefix">
            {bucket.value ? `${bucket.value}/${prefix.value}` : ""}
          </span>
          <input
            class="input-url"
            value={destName.value}
            placeholder="filename.dzip"
            onInput={(e: Event) =>
              (destName.value = (e.target as HTMLInputElement).value)
            }
          />
        </div>

        <div
          class="row"
          style="justify-content:flex-end;gap:6px;margin-top:2px"
        >
          <button class="btn-sm" onClick={onClose}>
            Cancel
          </button>
          <button
            class="btn"
            onClick={confirm}
            disabled={!bucket.value || !destName.value}
          >
            Select
          </button>
        </div>
      </div>
    </div>
  );
}
