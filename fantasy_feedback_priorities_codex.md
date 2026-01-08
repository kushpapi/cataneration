# Fantasy League Analytics App — Feedback & Priorities (for Codex)

_Last updated: 2026-01-06_

## Goal
Make the app tell the league story instantly:
1) **Trophies/Titles first**
2) **Rivalries + Head-to-Head context next**
3) **Season/behavior + deeper analytics last**

---

## All User Feedback (Grouped)

### A) Data gaps / prerequisites
- **Titles/championships not currently in data** — need to confirm: *is it in the API?* If not, derive from playoff bracket/matchups.

### B) Trophy Room + “Titles-first” landing experience
- Landing page: **first front-and-center visual should be # of titles**.
- Need a **main Trophy Room**: most trophies, points-for titles, “points for titles”.
- Add **position titles** (RB titles, WR titles, etc.) — likely requires player-level or positional points data.

### C) Head-to-Head Explorer (context + detail)
- Under “Context”, add:
  - **Playoff matchup record**
  - **# of blowouts**
  - **# of close games**
- Under “Context”, add a **dropdown/list of matchup results by date** (each historical matchup in the series).

### D) Filters (cross-cutting)
- Add **regular season vs playoffs** filter(s) on the **main table** (and propagate across relevant views).

### E) Weekly timeline view (storytelling)
- Add a **cumulative wins chart** (total wins over time) with a **line for each team**.
- Display **above the matchup breakout** section.

### F) Luck / unlucky + PF vs W/L discrepancy
- Add a metric/graph to show **lucky vs unlucky**, highlighting discrepancy between PF and W/L.
- Adjust visuals: **bad luck should be red with a down indicator**.
- Show how PF and W/L correlate (league + owner views if possible).

### G) Season-level / behavioral stats (optional)
- **# of trades per team per season** (interesting; may be out of scope depending on transactions data availability).

### H) “Consistency beyond titles”
- **# of top 3 playoff finishes**
- **# of top 3 PF seasons**

---

## Prioritization (P0 → P3)

### P0 — Ship first (highest impact + avoids rework)
1) **Confirm/implement Titles data source** (API vs derived).
2) **Landing page: Titles hero metric + titles leaderboard** (front-and-center).
3) **Global filters: Regular Season vs Playoffs** on main table (and propagate).

### P1 — Next (high delight, moderate effort)
4) **Head-to-head context:** playoff record + blowouts/close games.
5) **Head-to-head results-by-date dropdown/list** (series detail).
6) **Weekly timeline cumulative wins chart** (one line per team) above matchup breakout.
7) **Luck metric v1 + PF vs W/L view** + **bad luck red/down** semantics.

### P2 — Later (depth features)
8) Owner achievements: **Top-3 playoff finishes** + **Top-3 PF seasons**.

### P3 — Optional / scope-dependent
9) Trades per team per season.
10) Position titles + player-level deep dives (data-heavy).

---

## Proposed Definitions (Editable)

### Titles
- League championships (season champion).

### Playoff record
- H2H record restricted to weeks flagged as playoffs (or bracket-derived).

### Close games / Blowouts
- Close game: abs(margin) ≤ `CLOSE_MARGIN` (suggest 10; configurable)
- Blowout: abs(margin) ≥ `BLOWOUT_MARGIN` (suggest 30; configurable)

### Luck (v1 options)
Pick one based on data availability:
- **All-play win% – actual win%** (recommended if weekly scores exist)
- Expected wins model – actual wins
- PF rank vs W rank delta

---

## Data/API Questions (Must Answer)
1) Is there an API endpoint/field for season champion / titles?
2) Can we identify playoff weeks or bracket outcomes reliably?
3) Do we have transactions data (trades) by owner-season?
4) Can we obtain positional scoring or player-level scoring history?

---

## Estimate: Integrating Player-Level Data (for Position Titles)

### What “player-level” unlocks
- RB/WR/TE/QB/K/DST seasonal points by owner
- Position “titles” (e.g., “RB points champ”)
- Later: lineup efficiency, start/sit insights, roster construction stories

### Work required (typical)
- New grain tables:
  - `players`
  - `player_week` (player_id, week, points, position)
  - `lineup_week` (owner_id, week, player_id, slot, starter_flag)
  - derived: `owner_week_position_points`, `owner_season_position_points`
- Identity and history fixes:
  - stable player IDs across seasons
  - name/team changes
  - missing week backfills (if any)

### Rough effort ranges (solo dev)
- **Best case (clean API + stable IDs):** ~3–7 days
- **Normal case:** ~2–4 weeks
- **Hard case (incomplete history / scraping / inconsistent IDs):** 4–8+ weeks

### Biggest unknowns (drive the estimate)
- Do we have weekly player points directly, or must we reconstruct from box scores?
- Are player IDs stable back to 2013?
- Can we reliably distinguish starters vs bench and lineup slots?
- Are playoff weeks/brackets clearly indicated?

### Recommended approach
Start with **position totals by owner-season** (enables RB/WR titles) and defer full player-week drilldowns until the data layer is stable.

---

## Implementation Notes (Codex-ready)
- Keep thresholds configurable:
  - `CLOSE_MARGIN = 10`
  - `BLOWOUT_MARGIN = 30`
- Prefer marts layer:
  - `mart_titles`
  - `mart_head_to_head_context`
  - `mart_h2h_results_by_date`
  - `mart_weekly_cumulative_wins`
  - `mart_luck`
  - `mart_owner_achievements`
  - (optional) `mart_trades`

