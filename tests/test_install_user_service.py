from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.install_user_service import build_unit_text  # noqa: E402


def test_build_unit_text_rewrites_template_project_root(tmp_path):
    template = tmp_path / "listener.service"
    template.write_text(
        "\n".join(
            [
                "WorkingDirectory=/home/re/src/Listener",
                "ExecStart=/home/re/src/Listener/.venv/bin/python /home/re/src/Listener/main.py",
                "ExecReload=/home/re/src/Listener/.venv/bin/python /home/re/src/Listener/utils/listenerctl.py speech_gate_reset --reason systemd-reload",
                "ExecStop=-/home/re/src/Listener/.venv/bin/python /home/re/src/Listener/utils/listenerctl.py stop --reason systemd",
                "",
            ]
        ),
        encoding="utf-8",
    )
    project_root = tmp_path / "My Listener"

    unit_text = build_unit_text(project_root, template_path=template)

    assert "/home/re/src/Listener" not in unit_text
    assert f"WorkingDirectory={project_root.resolve()}" in unit_text
    assert f"{project_root.resolve()}/main.py" in unit_text
    assert f"{project_root.resolve()}/utils/listenerctl.py" in unit_text
    assert "ExecReload=" in unit_text
    assert "speech_gate_reset --reason systemd-reload" in unit_text
    assert "ExecStop=-" in unit_text
