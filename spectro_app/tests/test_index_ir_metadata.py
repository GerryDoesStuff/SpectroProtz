import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "index_ir_metadata.py"
    cmd = [sys.executable, str(script), *args]
    return subprocess.check_output(cmd, text=True)


def test_owner_continuation_and_field_filtering():
    output = run_cli("--fields", "title", "cas_registry_no", "molform", "owner")
    records = json.loads(output)
    target = next(
        (item for item in records if item["path"].endswith("C106309_IR_0.jdx")),
        None,
    )
    assert target is not None, "Expected metadata for C106309_IR_0.jdx"
    assert target["title"] == "Heptanoic acid, ethyl ester"
    assert target["cas_registry_no"] == "106-30-9"
    assert target["molform"] == "C 9 H 18 O 2"
    assert target["owner"].endswith("All rights reserved."), target["owner"]
    assert set(target.keys()) == {"path", "title", "cas_registry_no", "molform", "owner"}


def test_list_fields_contains_expected_entries():
    output = run_cli("--list-fields")
    fields = [line.strip() for line in output.splitlines() if line.strip()]
    for expected in {"title", "cas_registry_no", "molform", "owner"}:
        assert expected in fields
