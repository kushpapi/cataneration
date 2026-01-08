# Sleeper Playoffs Support

This document describes how playoff champion, playoff teams, and bracket seeding
are derived from Sleeper playoff endpoints.

## Endpoints

- Winners bracket: `GET https://api.sleeper.app/v1/league/{league_id}/winners_bracket`
- Losers bracket: `GET https://api.sleeper.app/v1/league/{league_id}/losers_bracket`

## Champion derivation

The champion is inferred from the placement matchup where `p == 1`.
If the winner field `w` is not populated yet, the champion is `None`.

## Playoff team derivation

Playoff teams are derived by scanning winners bracket entries and collecting
all concrete roster IDs found in:

- `t1` / `t2` (team slots)
- `w` / `l` (resolved winner/loser fields)

Placeholder objects (empty dicts) and missing values are ignored.
The result is a sorted list of unique roster IDs.

## Bracket seed map

Sleeper does not expose a canonical seed, so a bracket-slot-based mapping is used.
The earliest round (`r == min(r)`) is treated as the initial playoff round, and
slot keys are created without implying a seed rank:

- `R{r}_M{m}_T1` -> roster_id
- `R{r}_M{m}_T2` -> roster_id

Interpretation: slot labels indicate bracket position, not strength.
