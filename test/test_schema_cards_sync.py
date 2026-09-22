"""The assistant's parameter cards must match the live default config.

``chatbot/schema_cards.json`` is generated from ``_DEFAULT_CONFIG`` by
``dump_cards_json`` and feeds the assistant's RAG index. It had drifted before
(the timepoint-range cards were missing), so the assistant did not know about
settings the GUI had. Regenerate with::

    python -c "from behav3d.napari._assistant_schema import dump_cards_json; dump_cards_json('chatbot/schema_cards.json')"
"""
import json
from pathlib import Path

from behav3d.napari._assistant_schema import flatten_config_to_cards


def test_schema_cards_json_matches_default_config():
    path = Path(__file__).resolve().parents[1] / "chatbot" / "schema_cards.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    live = json.loads(json.dumps(flatten_config_to_cards(), default=str))
    assert on_disk == live, (
        "chatbot/schema_cards.json is out of date with _DEFAULT_CONFIG; regenerate it "
        "(see this test's module docstring)."
    )


def test_no_card_describes_a_removed_active_killing_setting():
    live = flatten_config_to_cards()
    keys = {c["key"] for c in live}
    for gone in ("observation_window", "death_signal_column", "killing_threshold_multiplier",
                 "absolute_killing_threshold", "use_absolute_threshold", "min_contact_duration",
                 "contact_column"):
        assert f"active_killing.{gone}" not in keys
    for present in ("target_cell_diameter_um", "causal_window_min", "attribution_radius_um"):
        assert f"active_killing.{present}" in keys
