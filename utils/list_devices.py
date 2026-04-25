import argparse
import json
import multiprocessing as mp
import queue
import shutil
import subprocess


MONITOR_MARKERS = (".monitor", " monitor", "monitor of")


def find_hostapi_index_by_name(apis, needle):
    needle = needle.lower()
    for idx, api in enumerate(apis):
        if needle in api.get("name", "").lower():
            return idx
    return None


def set_defaults_to_api(sd, apis, api_name):
    """Try to set sd.default.device to defaults of selected host API."""
    idx = find_hostapi_index_by_name(apis, api_name)
    if idx is None:
        return False, f"Host API '{api_name}' not found."
    api = apis[idx]
    di = api.get("default_input_device", -1)
    do = api.get("default_output_device", -1)
    if di == -1 and do == -1:
        return False, f"'{api['name']}' has no default devices."
    cur_in, cur_out = sd.default.device
    if di != -1:
        cur_in = di
    if do != -1:
        cur_out = do
    sd.default.device = (cur_in, cur_out)
    return True, f"sd.default.device -> (in={cur_in}, out={cur_out}) from '{api['name']}'"


def is_monitor_device(name, hostapi_name):
    haystack = f"{name} {hostapi_name}".lower()
    return any(marker in haystack for marker in MONITOR_MARKERS)


def collect_devices(apis, devices, default_device):
    d_in, d_out = default_device
    rows = []
    for i, inf in enumerate(devices):
        hostapi_idx = inf["hostapi"]
        hostapi_name = apis[hostapi_idx]["name"]
        max_in = int(inf.get("max_input_channels", 0))
        max_out = int(inf.get("max_output_channels", 0))
        name = inf["name"]
        monitor = is_monitor_device(name, hostapi_name) and max_in > 0

        pa_default = []
        if d_in is not None and i == d_in:
            pa_default.append("mic-in")
        if d_out is not None and i == d_out:
            pa_default.append("speaker-out")

        api_default = []
        if i == apis[hostapi_idx].get("default_input_device", -1):
            api_default.append("IN*")
        if i == apis[hostapi_idx].get("default_output_device", -1):
            api_default.append("OUT*")

        rows.append(
            {
                "id": i,
                "name": name,
                "host_api": hostapi_name,
                "max_input_channels": max_in,
                "max_output_channels": max_out,
                "default_samplerate": int(inf.get("default_samplerate", 0)),
                "pa_default": " ".join(pa_default),
                "api_default": " ".join(api_default),
                "is_monitor": monitor,
            }
        )
    return rows


def filter_rows(rows, *, input_only=False, output_only=False, monitors=False):
    result = []
    for row in rows:
        if input_only and row["max_input_channels"] <= 0:
            continue
        if output_only and row["max_output_channels"] <= 0:
            continue
        if monitors and not row["is_monitor"]:
            continue
        result.append(row)
    return result


def filter_pulse_sources(sources: list[dict], *, monitors=False) -> list[dict]:
    if not monitors:
        return list(sources)
    return [source for source in sources if source["is_monitor"]]


def print_table(rows):
    terminal_width = shutil.get_terminal_size((120, 24)).columns
    fixed_width = 3 + 2 + 12 + 2 + 4 + 2 + 4 + 2 + 6 + 2 + 12 + 2 + 8 + 2 + 3
    name_width = max(28, min(64, terminal_width - fixed_width))
    columns = [
        ("ID", 3, "id", ">"),
        ("Name", name_width, "name", "<"),
        ("Host API", 12, "host_api", "<"),
        ("In", 4, "max_input_channels", ">"),
        ("Out", 4, "max_output_channels", ">"),
        ("SR", 6, "default_samplerate", ">"),
        ("Default", 12, "pa_default", "<"),
        ("API def", 8, "api_default", "<"),
        ("Mon", 3, "monitor_label", "<"),
    ]

    def cell(value, width, align="<"):
        text = str(value)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        if align == ">":
            return text.rjust(width)
        return text.ljust(width)

    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["monitor_label"] = "yes" if row["is_monitor"] else ""
        normalized_rows.append(normalized)

    header = "  ".join(cell(title, width, "<") for title, width, _key, _align in columns)
    rule = "  ".join("-" * width for _title, width, _key, _align in columns)
    print(header)
    print(rule)
    for row in normalized_rows:
        print(
            "  ".join(
                cell(row.get(key, ""), width, align)
                for _title, width, key, align in columns
            )
        )


def query_pulse_sources() -> tuple[list[dict], str | None]:
    try:
        default_proc = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        default_source = default_proc.stdout.strip() if default_proc.returncode == 0 else None
        proc = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return [], None
    if proc.returncode != 0:
        return [], default_source

    sources = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        index, name, driver, sample_spec, state = parts[:5]
        sources.append(
            {
                "id": index,
                "kind": "monitor" if ".monitor" in name.lower() else "source",
                "name": name,
                "driver": driver,
                "sample_spec": sample_spec,
                "state": state,
                "is_default": name == default_source,
                "is_monitor": ".monitor" in name.lower(),
            }
        )
    return sources, default_source


def print_pulse_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print("\nPipeWire/Pulse sources")
    terminal_width = shutil.get_terminal_size((120, 24)).columns
    fixed_width = 4 + 2 + 8 + 2 + 12 + 2 + 20 + 2 + 10 + 2 + 3 + 2 + 3
    name_width = max(36, min(82, terminal_width - fixed_width))
    columns = [
        ("ID", 4, "id", ">"),
        ("Kind", 8, "kind", "<"),
        ("Source name", name_width, "name", "<"),
        ("Driver", 12, "driver", "<"),
        ("Format", 20, "sample_spec", "<"),
        ("State", 10, "state", "<"),
        ("Def", 3, "default_label", "<"),
        ("Mon", 3, "monitor_label", "<"),
    ]

    def cell(value, width, align="<"):
        text = str(value)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        return text.rjust(width) if align == ">" else text.ljust(width)

    header = "  ".join(cell(title, width) for title, width, _key, _align in columns)
    rule = "  ".join("-" * width for _title, width, _key, _align in columns)
    print(header)
    print(rule)
    for source in sources:
        row = dict(source)
        row["default_label"] = "yes" if source["is_default"] else ""
        row["monitor_label"] = "yes" if source["is_monitor"] else ""
        print(
            "  ".join(
                cell(row.get(key, ""), width, align)
                for _title, width, key, align in columns
            )
        )


def query_sound_devices(prefer_api=None, timeout_s=5.0):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_query_sound_devices_worker, args=(prefer_api, result_queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        raise TimeoutError(
            f"sounddevice did not respond within {timeout_s:.1f}s while querying devices"
        )
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        raise RuntimeError("sounddevice query failed without details")
    if status == "error":
        raise RuntimeError(str(payload))
    return payload


def _query_sound_devices_worker(prefer_api, result_queue):
    try:
        import sounddevice as sd

        apis = [dict(api) for api in sd.query_hostapis()]
        note = None
        if prefer_api:
            _ok, note = set_defaults_to_api(sd, apis, prefer_api)
        devices = [dict(device) for device in sd.query_devices()]
        default_device = tuple(sd.default.device)
        result_queue.put(
            (
                "ok",
                {
                    "apis": apis,
                    "devices": devices,
                    "default_device": default_device,
                    "note": note,
                },
            )
        )
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefer-api",
        type=str,
        default=None,
        help="Select default devices from this Host API (e.g. wasapi, mme, directsound, wdm, asio, pulse, pipewire)",
    )
    p.add_argument("--json", action="store_true", help="Print devices as JSON.")
    p.add_argument("--input-only", action="store_true", help="Show only input-capable devices.")
    p.add_argument("--output-only", action="store_true", help="Show only output-capable devices.")
    p.add_argument(
        "--monitors",
        action="store_true",
        help="Show likely PulseAudio/PipeWire monitor sources for Linux loopback.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for sounddevice before failing cleanly.",
    )
    args = p.parse_args()

    try:
        query = query_sound_devices(prefer_api=args.prefer_api, timeout_s=args.timeout)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"Failed to query audio devices: {exc}")
        return 1

    rows = collect_devices(query["apis"], query["devices"], query["default_device"])
    rows = filter_rows(
        rows,
        input_only=args.input_only,
        output_only=args.output_only,
        monitors=args.monitors,
    )
    pulse_sources, default_pulse_source = query_pulse_sources()
    pulse_sources = filter_pulse_sources(pulse_sources, monitors=args.monitors)

    if args.json:
        print(
            json.dumps(
                {
                    "sounddevice_devices": rows,
                    "pulse_sources": pulse_sources,
                    "default_pulse_source": default_pulse_source,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if rows:
            print_table(rows)
        elif not pulse_sources:
            print_table(rows)
        note = query.get("note")
        if note:
            print("\nNote:", note)
        print_pulse_sources(pulse_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
