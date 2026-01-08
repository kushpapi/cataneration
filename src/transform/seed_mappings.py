"""Generate owner mapping todo files from staging data."""

from pathlib import Path
import pandas as pd


def seed_team_aliases() -> None:
    """
    Generate team_aliases.todo.csv for unmapped teams.

    Scans all staging tables and identifies teams without owner mappings.
    """
    # Load existing mappings
    team_aliases_path = Path("assets/team_aliases.csv")
    existing_aliases = pd.read_csv(team_aliases_path)

    # Collect all teams from staging
    staging_dir = Path("data/staging")
    all_teams = []

    for staging_file in staging_dir.rglob("stg_teams.csv"):
        # Extract platform and season from path: data/staging/{platform}/{season}/
        parts = staging_file.parts
        platform = parts[2]
        season = int(parts[3])

        teams_df = pd.read_csv(staging_file)
        teams_df["platform"] = platform
        teams_df["season"] = season
        all_teams.append(teams_df[["platform", "season", "team_name"]])

    if not all_teams:
        print("No staging data found")
        return

    # Combine all teams
    all_teams_df = pd.concat(all_teams, ignore_index=True)
    all_teams_df = all_teams_df.drop_duplicates()

    # Find unmapped teams
    unmapped = all_teams_df.merge(
        existing_aliases,
        on=["platform", "season", "team_name"],
        how="left",
        indicator=True
    )
    unmapped = unmapped[unmapped["_merge"] == "left_only"]
    unmapped = unmapped[["platform", "season", "team_name"]]
    unmapped["owner_id"] = ""  # Empty column for manual filling

    if unmapped.empty:
        print("✓ All teams are mapped")
        return

    # Write todo file
    todo_path = Path("assets/team_aliases.todo.csv")
    unmapped.to_csv(todo_path, index=False)
    print(f"✓ Generated {todo_path} with {len(unmapped)} unmapped teams")
    print(f"  → Fill in owner_id column, then move to team_aliases.csv")


def seed_owner_aliases() -> None:
    """
    Generate owner_aliases.todo.csv for unmapped platform user IDs.

    Note: MFL doesn't expose user IDs in franchise data, so this is a placeholder
    for future Fleaflicker/Sleeper implementation.
    """
    owner_aliases_path = Path("assets/owner_aliases.csv")
    existing_aliases = pd.read_csv(owner_aliases_path)

    # TODO: Implement when we have platforms with user IDs (Fleaflicker, Sleeper)
    print("✓ Owner aliases seeding skipped (MFL uses team names)")


def seed_all_mappings() -> None:
    """Seed all mapping todo files."""
    print("Seeding owner mappings...")
    seed_team_aliases()
    seed_owner_aliases()
