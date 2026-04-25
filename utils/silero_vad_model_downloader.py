import argparse
import os


def download_silero_vad(dst: str, *, force_reload: bool = False) -> str:
    import torch

    torch.set_num_threads(1)
    model, _utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=force_reload,
    )

    jit_obj = None
    for attr in (model, getattr(model, "model", None), getattr(model, "_model", None)):
        try:
            import torch.jit

            if isinstance(attr, torch.jit.ScriptModule) or type(attr).__name__ in (
                "RecursiveScriptModule",
                "ScriptModule",
            ):
                jit_obj = attr
                break
        except Exception:
            pass

    if not jit_obj:
        raise RuntimeError(
            "Could not find ScriptModule inside hub model. "
            "Use weights from Releases if torch.hub wrapper changes."
        )

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.jit.save(jit_obj, dst)
    return dst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download and save Silero VAD JIT model.")
    parser.add_argument("--out", default=os.path.join("models", "silero_vad_v6.jit"))
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check imports/CLI without downloading the model.",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        import torch

        print(f"self-test ok: torch={torch.__version__} out={args.out}")
        return 0

    dst = download_silero_vad(args.out, force_reload=args.force_reload)
    print(f"Saved -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
