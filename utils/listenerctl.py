#!/usr/bin/env python3
"""CLI for Listener's local control API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_CONTROL_URL = "http://127.0.0.1:18790"
SPEECH_GATE_MODES = ["normal", "mute", "chatty", "standby"]


def add_set_mode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ttl", type=float, default=None, help="Temporary mode TTL in seconds.")
    parser.add_argument("--reason", default="", help="Human-readable reason.")
    parser.add_argument("--source", default="listenerctl", help="Mode-change source label.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response.")


def add_reason_source_json_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reason", default="", help="Human-readable reason.")
    parser.add_argument("--source", default="listenerctl", help="Change source label.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control a running Listener process.")
    parser.add_argument(
        "--url",
        default=os.environ.get("LISTENER_CONTROL_URL", DEFAULT_CONTROL_URL),
        help="Listener control API base URL.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LISTENER_CONTROL_TOKEN"),
        help="Bearer token for Listener control API.",
    )
    subparsers = parser.add_subparsers(dest="resource", required=True)

    speech_gate = subparsers.add_parser("speech-gate", help="Control SpeechGate.")
    speech_gate_subparsers = speech_gate.add_subparsers(dest="action", required=True)

    status = speech_gate_subparsers.add_parser("status", help="Show SpeechGate status.")
    status.add_argument("--json", action="store_true", help="Print raw JSON response.")

    set_mode = speech_gate_subparsers.add_parser("set-mode", help="Set SpeechGate mode.")
    set_mode.add_argument("mode", choices=SPEECH_GATE_MODES)
    add_set_mode_options(set_mode)

    reset = speech_gate_subparsers.add_parser(
        "reset",
        help="Recover normal speech gate mode, speaker output, and ducking state.",
    )
    add_reason_source_json_options(reset)

    status_alias = subparsers.add_parser("status", help="Shortcut for: speech-gate status.")
    status_alias.add_argument("--json", action="store_true", help="Print raw JSON response.")
    status_alias.set_defaults(resource="speech-gate", action="status")

    for mode in SPEECH_GATE_MODES:
        mode_alias = subparsers.add_parser(
            mode,
            help=f"Shortcut for: speech-gate set-mode {mode}.",
        )
        add_set_mode_options(mode_alias)
        mode_alias.set_defaults(resource="speech-gate", action="set-mode", mode=mode)

    reset_alias = subparsers.add_parser(
        "speech-gate-reset",
        help="Shortcut for: speech-gate reset.",
    )
    add_reason_source_json_options(reset_alias)
    reset_alias.set_defaults(resource="speech-gate", action="reset")

    reset_alias_underscore = subparsers.add_parser(
        "speech_gate_reset",
        help="Shortcut for: speech-gate reset.",
    )
    add_reason_source_json_options(reset_alias_underscore)
    reset_alias_underscore.set_defaults(resource="speech-gate", action="reset")

    speaker = subparsers.add_parser("speaker", help="Control integrated Speaker playback.")
    speaker_subparsers = speaker.add_subparsers(dest="action", required=True)

    speaker_status = speaker_subparsers.add_parser("status", help="Show Speaker status.")
    speaker_status.add_argument("--json", action="store_true", help="Print raw JSON response.")

    for action, enabled in (("on", True), ("off", False)):
        speaker_mode = speaker_subparsers.add_parser(action, help=f"Turn Speaker {action}.")
        add_reason_source_json_options(speaker_mode)
        speaker_mode.set_defaults(enabled=enabled)

    voice_status = subparsers.add_parser("voice-status", help="Shortcut for: speaker status.")
    voice_status.add_argument("--json", action="store_true", help="Print raw JSON response.")
    voice_status.set_defaults(resource="speaker", action="status")

    for alias, enabled in (("voice-on", True), ("voice-off", False)):
        voice_alias = subparsers.add_parser(alias, help=f"Shortcut for: speaker {'on' if enabled else 'off'}.")
        add_reason_source_json_options(voice_alias)
        voice_alias.set_defaults(resource="speaker", action="enabled", enabled=enabled)

    health = subparsers.add_parser("health", help="Check Listener control API liveness.")
    health.add_argument("--json", action="store_true", help="Print raw JSON response.")
    health.set_defaults(resource="service", action="health")

    ready = subparsers.add_parser("ready", help="Check Listener service readiness.")
    ready.add_argument("--json", action="store_true", help="Print raw JSON response.")
    ready.set_defaults(resource="service", action="ready")

    stop = subparsers.add_parser("stop", help="Ask Listener to stop gracefully.")
    stop.add_argument("--reason", default="listenerctl", help="Human-readable reason.")
    stop.add_argument("--json", action="store_true", help="Print raw JSON response.")
    stop.set_defaults(resource="service", action="stop")

    return parser


def build_set_mode_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": args.mode,
        "source": args.source,
        "reason": args.reason,
    }
    if args.ttl is not None:
        payload["ttl_seconds"] = args.ttl
    return payload


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:  # noqa: S310 - local user-controlled URL.
            data = response.read()
            return response.status, _decode_json(data)
    except HTTPError as exc:
        return exc.code, _decode_json(exc.read())
    except URLError as exc:
        return 0, {"ok": False, "error": f"connection_failed: {exc.reason}"}


def _decode_json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8") if data else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "error": "invalid_json_response"}
    return value if isinstance(value, dict) else {"ok": False, "error": "invalid_json_response"}


def format_speech_gate_status(speech_gate: dict[str, Any]) -> str:
    mode = speech_gate.get("mode", "unknown")
    temporary = bool(speech_gate.get("temporary"))
    state = "temporary" if temporary else "permanent"
    parts = [f"speech_gate mode={mode}", f"state={state}"]

    if temporary:
        expires_in = _format_seconds(speech_gate.get("expires_in_seconds"))
        expires_at = _format_timestamp(speech_gate.get("expires_at"))
        restore = speech_gate.get("restore_mode") or "-"
        parts.extend(
            [
                f"expires_in={expires_in}",
                f"expires_at={expires_at}",
                f"restore={restore}",
            ]
        )
    else:
        parts.extend(["expires_in=-", "expires_at=-", "restore=-"])

    source = speech_gate.get("source")
    if source:
        parts.append(f"source={source}")
    reason = str(speech_gate.get("reason") or "")
    if reason:
        parts.append(f"reason={json.dumps(reason, ensure_ascii=False)}")
    return " ".join(parts)


def format_speaker_status(speaker: dict[str, Any]) -> str:
    enabled = "on" if bool(speaker.get("enabled")) else "off"
    running = "running" if bool(speaker.get("running")) else "stopped"
    connected = "connected" if bool(speaker.get("connected")) else "disconnected"
    playback = speaker.get("playback") if isinstance(speaker.get("playback"), dict) else {}
    parts = [
        f"speaker={enabled}",
        f"agent={running}",
        f"gateway={connected}",
        f"mode={speaker.get('mode', 'unknown')}",
        f"session={speaker.get('session_key', '-')}",
        f"queue={playback.get('queue_size', 0)}",
    ]
    current = playback.get("current")
    if current:
        parts.append(f"current={current}")
    reason = playback.get("last_interrupt_reason")
    if reason:
        parts.append(f"last_interrupt={json.dumps(str(reason), ensure_ascii=False)}")
    error = speaker.get("last_error")
    if error:
        parts.append(f"error={json.dumps(str(error), ensure_ascii=False)}")
    return " ".join(parts)


def format_ready_status(data: dict[str, Any]) -> str:
    state = "ready" if bool(data.get("ready")) else "not_ready"
    parts = [f"listener={state}"]
    components = data.get("components")
    if isinstance(components, dict):
        component_parts = []
        for name, component in components.items():
            if not isinstance(component, dict):
                continue
            component_state = str(component.get("state") or "unknown")
            suffix = "!" if component.get("critical") and not component.get("ok") else ""
            component_parts.append(f"{name}={component_state}{suffix}")
        if component_parts:
            parts.append("components=" + ",".join(component_parts))
    last_error = data.get("last_error")
    if last_error:
        parts.append(f"last_error={json.dumps(str(last_error), ensure_ascii=False)}")
    return " ".join(parts)


def format_speech_gate_reset_status(data: dict[str, Any]) -> str:
    parts = ["speech_gate_reset=ok"]
    speech_gate = data.get("speech_gate")
    if isinstance(speech_gate, dict):
        parts.append(f"mode={speech_gate.get('mode', 'unknown')}")
    speaker = data.get("speaker")
    if isinstance(speaker, dict):
        parts.append(f"speaker={'on' if bool(speaker.get('enabled')) else 'off'}")
    ducking = data.get("ducking")
    if isinstance(ducking, dict):
        restored_ids = ducking.get("restored_sink_input_ids")
        missing_ids = ducking.get("missing_sink_input_ids")
        route_keys = ducking.get("listener_route_keys")
        if isinstance(restored_ids, list):
            parts.append(f"restored={len(restored_ids)}")
        if isinstance(route_keys, list) and route_keys:
            parts.append(f"routes={len(route_keys)}")
        if isinstance(missing_ids, list) and missing_ids:
            parts.append(f"missing={len(missing_ids)}")
    dropped = data.get("dropped")
    if dropped not in (None, ""):
        parts.append(f"dropped={dropped}")
    return " ".join(parts)


def _format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "-"


def _format_timestamp(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return "-"
    if timestamp <= 0:
        return "-"
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def _print_response(data: dict[str, Any], *, raw_json: bool) -> None:
    if raw_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not data.get("ok"):
        print(f"error: {data.get('error', 'unknown_error')}", file=sys.stderr)
        return
    speech_gate = data.get("speech_gate")
    if not isinstance(speech_gate, dict):
        print("ok")
        return
    print(format_speech_gate_status(speech_gate))


def _print_speaker_response(data: dict[str, Any], *, raw_json: bool) -> None:
    if raw_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not data.get("ok"):
        print(f"error: {data.get('error', 'unknown_error')}", file=sys.stderr)
        return
    speaker = data.get("speaker")
    if not isinstance(speaker, dict):
        print("ok")
        return
    print(format_speaker_status(speaker))


def _print_service_response(data: dict[str, Any], *, raw_json: bool, ready: bool = False) -> None:
    if raw_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not data.get("ok"):
        print(f"error: {data.get('error', 'unknown_error')}", file=sys.stderr)
        return
    if ready:
        print(format_ready_status(data))
        return
    if data.get("stopping"):
        print("listener stopping")
        return
    print("listener healthy")


def _print_reset_response(data: dict[str, Any], *, raw_json: bool) -> None:
    if raw_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if not data.get("ok"):
        print(f"error: {data.get('error', 'unknown_error')}", file=sys.stderr)
        return
    print(format_speech_gate_reset_status(data))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resource == "speech-gate" and args.action == "status":
        status, data = request_json(args.url, "/speech-gate/status", token=args.token)
        _print_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "speech-gate" and args.action == "set-mode":
        payload = build_set_mode_payload(args)
        status, data = request_json(
            args.url,
            "/speech-gate/mode",
            method="POST",
            token=args.token,
            payload=payload,
        )
        _print_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "speech-gate" and args.action == "reset":
        status, data = request_json(
            args.url,
            "/speech-gate/reset",
            method="POST",
            token=args.token,
            payload={
                "source": args.source,
                "reason": args.reason or "speech_gate_reset",
            },
        )
        _print_reset_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "speaker" and args.action == "status":
        status, data = request_json(args.url, "/speaker/status", token=args.token)
        _print_speaker_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "speaker" and args.action in {"on", "off", "enabled"}:
        enabled = bool(args.enabled)
        status, data = request_json(
            args.url,
            "/speaker/enabled",
            method="POST",
            token=args.token,
            payload={
                "enabled": enabled,
                "source": args.source,
                "reason": args.reason,
            },
        )
        _print_speaker_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "service" and args.action == "health":
        status, data = request_json(args.url, "/health", token=args.token)
        _print_service_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    if args.resource == "service" and args.action == "ready":
        status, data = request_json(args.url, "/ready", token=args.token)
        _print_service_response(data, raw_json=args.json, ready=True)
        return 0 if 200 <= status < 300 and data.get("ok") and data.get("ready") else 1

    if args.resource == "service" and args.action == "stop":
        status, data = request_json(
            args.url,
            "/shutdown",
            method="POST",
            token=args.token,
            payload={"reason": args.reason},
        )
        _print_service_response(data, raw_json=args.json)
        return 0 if 200 <= status < 300 and data.get("ok") else 1

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
