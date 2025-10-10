from dataclasses import replace

from spectro_app.ui.docks.file_queue_models import QueueEntry, order_queue_entries


def _names(entries):
    return [entry.display_name for entry in entries]


def test_order_queue_entries_prioritizes_blank_metadata():
    entries = [
        QueueEntry(path="sample-1", display_name="Sample 1", role="sample"),
        QueueEntry(
            path="blank-1",
            display_name="Blank 1",
            metadata={"role": "blank"},
        ),
        QueueEntry(
            path="reference-1",
            display_name="Reference 1",
            metadata={"type": "Reference"},
        ),
        QueueEntry(path="sample-2", display_name="Sample 2", role="sample"),
    ]

    ordered = order_queue_entries(entries)

    assert _names(ordered) == [
        "Blank 1",
        "Reference 1",
        "Sample 1",
        "Sample 2",
    ]


def test_order_queue_entries_respects_blank_overrides():
    entries = [
        QueueEntry(path="sample-1", display_name="Sample 1", role="sample"),
        QueueEntry(path="sample-2", display_name="Sample 2", role="sample"),
        QueueEntry(path="sample-3", display_name="Sample 3", role="sample"),
    ]

    overridden = [
        entries[0],
        replace(
            entries[1],
            role="blank",
            metadata={"role": "blank"},
            overrides={"role": "blank"},
        ),
        entries[2],
    ]

    ordered = order_queue_entries(overridden)

    assert _names(ordered) == [
        "Sample 2",
        "Sample 1",
        "Sample 3",
    ]

    # Ensure the override entry maintains its overrides after reordering.
    assert ordered[0].overrides.get("role") == "blank"
