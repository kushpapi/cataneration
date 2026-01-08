import json
from pathlib import Path

from src.common.sleeper_playoffs import (
    derive_bracket_seed_map,
    derive_champion_roster_id,
    derive_playoff_roster_ids,
)


def load_fixture() -> list:
    fixture_path = Path("tests/fixtures/sleeper_winners_bracket.json")
    return json.loads(fixture_path.read_text())


def test_derive_champion_roster_id_finished():
    bracket = load_fixture()
    assert derive_champion_roster_id(bracket) == 4


def test_derive_champion_roster_id_unfinished():
    bracket = load_fixture()
    for matchup in bracket:
        if matchup.get("p") == 1:
            matchup["w"] = None
    assert derive_champion_roster_id(bracket) is None


def test_derive_playoff_roster_ids():
    bracket = load_fixture()
    roster_ids = derive_playoff_roster_ids(bracket)
    assert roster_ids == [1, 2, 3, 4, 5, 6]


def test_derive_bracket_seed_map():
    bracket = load_fixture()
    seed_map = derive_bracket_seed_map(bracket)
    assert seed_map == {
        "R1_M1_T1": 1,
        "R1_M1_T2": 2,
        "R1_M2_T1": 3,
        "R1_M2_T2": 4,
        "R1_M3_T1": 5,
        "R1_M4_T2": 6,
    }
