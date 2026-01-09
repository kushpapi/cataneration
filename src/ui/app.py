#!/usr/bin/env python3
"""
Cataneration Trophy Room
Bloomberg & The Athletic Editorial Standards Implementation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ==============================================================================
# PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Cataneration Trophy Room",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg-inspired color palette
COLORS = {
    'primary': '#FF9500',      # Bloomberg orange
    'secondary': '#119DFF',    # Bloomberg blue
    'success': '#00B271',      # Green for positive
    'danger': '#FF3B30',       # Red for negative
    'text_dark': '#000000',    # Black for maximum readability
    'text_light': '#4A4A4A',   # Darker gray for better contrast
    'background': '#F5F5F7',   # Light gray
    'accent': '#666666'        # Darker medium gray
}

CLOSE_MARGIN = 10
BLOWOUT_MARGIN = 30

# Custom CSS for professional typography and spacing
st.markdown("""
<style>
    /* Bloomberg-inspired typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Improve metric displays */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #000000;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #2A2A2A;
    }

    /* Clean dataframe styling */
    .dataframe {
        font-size: 0.95rem;
        color: #000000;
    }

    .dataframe thead th {
        background-color: #F5F5F7;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        color: #1A1A1A;
    }

    .dataframe tbody td {
        color: #000000;
    }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: #F5F5F7;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #000000;
    }

    /* Button styling */
    .stButton>button {
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.2s;
        color: #000000;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
    }

    .stTabs [data-baseweb="tab"] {
        color: #000000;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: #FF9500;
        font-weight: 600;
    }

    /* ========================================
       COMPREHENSIVE DROPDOWN FIXES
       ======================================== */

    /* Select box labels */
    .stSelectbox label, .stTextInput label {
        color: #000000 !important;
        font-weight: 600;
    }

    /* Main select container - white background */
    .stSelectbox > div > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #E5E5E7 !important;
    }

    /* Selected value display - black text on white */
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #000000 !important;
    }

    /* Dropdown menu container */
    [data-baseweb="popover"],
    [role="listbox"],
    ul[role="listbox"] {
        background-color: white !important;
        border: 1px solid #E5E5E7 !important;
    }

    /* Dropdown options - white bg, black text */
    [data-baseweb="select"] [role="option"],
    [role="listbox"] > li,
    ul[role="listbox"] > li {
        background-color: white !important;
        color: #000000 !important;
    }

    /* Dropdown option hover - orange bg, white text */
    [data-baseweb="select"] [role="option"]:hover,
    [role="listbox"] > li:hover,
    ul[role="listbox"] > li:hover {
        background-color: #FF9500 !important;
        color: white !important;
    }

    /* Active/selected option in list - light orange bg, black text */
    [data-baseweb="select"] [role="option"][aria-selected="true"],
    [role="listbox"] > li[aria-selected="true"],
    ul[role="listbox"] > li[aria-selected="true"] {
        background-color: #FFE5CC !important;
        color: #000000 !important;
        font-weight: 600;
    }

    /* Override any text color in options */
    [role="option"] span,
    [role="option"] div,
    [role="listbox"] span,
    [role="listbox"] div {
        color: inherit !important;
    }

    /* Ensure placeholder text is visible */
    [data-baseweb="select"] input::placeholder {
        color: #666666 !important;
    }

    /* Selectbox when focused */
    .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #FF9500 !important;
        box-shadow: 0 0 0 1px #FF9500 !important;
    }

    /* Additional aggressive overrides for Streamlit defaults */
    div[class*="st-"] div[data-baseweb="select"],
    div[class*="st-"] [role="listbox"],
    div[class*="st-"] [role="option"] {
        background-color: white !important;
    }

    div[class*="st-"] div[data-baseweb="select"] *,
    div[class*="st-"] [role="listbox"] *,
    div[class*="st-"] [role="option"] * {
        color: #000000 !important;
    }

    /* Override hover states on all elements */
    div[class*="st-"] [role="option"]:hover,
    div[class*="st-"] [role="option"]:hover * {
        background-color: #FF9500 !important;
        color: white !important;
    }

    /* Radio buttons (if used in dropdowns) */
    .stRadio label {
        color: #000000 !important;
    }

    /* NUCLEAR OPTION: Override everything in selectbox */
    [data-testid="stSelectbox"],
    [data-testid="stSelectbox"] *:not(svg):not(path) {
        background-color: white !important;
        color: #000000 !important;
    }

    [data-testid="stSelectbox"] [role="listbox"],
    [data-testid="stSelectbox"] [role="listbox"] * {
        background-color: white !important;
        color: #000000 !important;
    }

    [data-testid="stSelectbox"] [role="option"]:hover,
    [data-testid="stSelectbox"] [role="option"]:hover * {
        background-color: #FF9500 !important;
        color: white !important;
    }

    /* Metric delta colors - restore green/red */
    [data-testid="stMetricDelta"] svg {
        fill: currentColor;
    }

    /* Positive deltas - green */
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {
        color: #00B271 !important;
    }

    /* Negative deltas - red */
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
        color: #FF3B30 !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.875rem;
    }

    /* Make sure delta text is visible */
    [data-testid="stMetricDelta"] span {
        color: inherit;
    }

    /* Title styling */
    h1 {
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.5rem;
    }

    h2, h3 {
        font-weight: 600;
        color: #000000;
    }

    /* Caption improvements */
    .caption {
        color: #3A3A3A;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Main content area - ensure white background */
    .main .block-container {
        background-color: white;
        padding: 2rem 1rem;
    }

    /* All paragraph text */
    p, div, span, label {
        color: #000000;
    }

    /* Streamlit caption elements */
    [data-testid="stCaptionContainer"] {
        color: #3A3A3A !important;
    }

    /* Ensure main content background is white */
    [data-testid="stAppViewContainer"] {
        background-color: white;
    }

    /* Force white background on main content */
    section[data-testid="stAppViewContainer"] > div {
        background-color: white;
    }

    /* Markdown text in main area */
    .main p, .main span, .main div {
        color: #000000 !important;
    }

    /* Footer styling */
    .footer {
        font-size: 0.75rem;
        color: #4A4A4A;
        padding-top: 2rem;
        border-top: 1px solid #E5E5E7;
        margin-top: 3rem;
        font-weight: 500;
    }

    .trophy-hero {
        background: linear-gradient(135deg, #FFF4D6 0%, #F7E7B0 100%);
        border: 1px solid #E5D7A3;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        text-align: center;
    }

    .trophy-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #1A1A1A;
    }

    .trophy-subtitle {
        font-size: 1.05rem;
        color: #4A4A4A;
        margin-top: 0.25rem;
    }

    .trophy-micro {
        font-size: 0.8rem;
        color: #6B6B6B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

@st.cache_data
def load_all_time_stats() -> pd.DataFrame:
    """Load all-time owner statistics."""
    df = pd.read_csv("data/mart/mart_owner_all_time.csv")
    # Sort by win percentage
    df = df.sort_values('win_pct', ascending=False).reset_index(drop=True)
    return df

@st.cache_data
def load_season_stats() -> pd.DataFrame:
    """Load season-by-season owner statistics."""
    return pd.read_csv("data/mart/mart_owner_season.csv")

@st.cache_data
def load_h2h_stats() -> pd.DataFrame:
    """Load head-to-head matchup statistics."""
    return pd.read_csv("data/mart/mart_h2h.csv")

@st.cache_data
def load_matchups() -> pd.DataFrame:
    """Load all matchup results."""
    return pd.read_csv("data/mart/mart_matchups.csv")

@st.cache_data
def load_owner_achievements() -> pd.DataFrame:
    """Load owner achievements (top-3 playoff finishes and PF seasons)."""
    try:
        return pd.read_csv("data/mart/mart_owner_achievements.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "owner_id",
            "top3_playoff_finishes",
            "top3_playoff_seasons",
            "top3_pf_seasons",
            "top3_pf_seasons_list",
            "top3_winners_finishes",
            "top3_winners_seasons",
            "top2_seeds",
            "top2_seeds_seasons"
        ])

@st.cache_data
def load_titles() -> pd.DataFrame:
    """Load season titles (champion + #1 loser)."""
    try:
        return pd.read_csv("data/mart/mart_titles.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "season",
            "platform",
            "platform_league_id",
            "champion_owner_id",
            "champion_team_name",
            "losers_champ_owner_id",
            "losers_champ_team_name"
        ])

@st.cache_data
def load_playoff_seeds() -> pd.DataFrame:
    """Load playoff bracket seed slots."""
    try:
        return pd.read_csv("data/mart/mart_playoff_seeds.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "season",
            "platform",
            "platform_league_id",
            "bracket_type",
            "round",
            "matchup_id",
            "slot",
            "platform_team_id",
            "owner_id",
            "team_name",
            "seed"
        ])

@st.cache_data
def load_owners() -> pd.DataFrame:
    """Load canonical owner display names."""
    try:
        return pd.read_csv("assets/owners.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["owner_id", "display_name"])

@st.cache_data
def load_team_aliases() -> pd.DataFrame:
    """Load team name to owner mappings by season/platform."""
    try:
        return pd.read_csv("assets/team_aliases.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["platform", "season", "team_name", "owner_id"])

@st.cache_data
def get_owner_display_map() -> dict:
    """Map owner_id -> latest franchise name, falling back to display_name/owner_id."""
    owners_df = load_owners()
    display_map = {
        row["owner_id"]: row["display_name"]
        for _, row in owners_df.iterrows()
        if isinstance(row.get("owner_id"), str)
    }
    aliases = load_team_aliases()
    if aliases.empty:
        return display_map

    platform_priority = {"sleeper": 3, "fleaflicker": 2, "mfl": 1}
    aliases = aliases.dropna(subset=["owner_id", "team_name", "season"]).copy()
    aliases["platform"] = aliases["platform"].astype(str).str.lower()
    aliases["platform_priority"] = aliases["platform"].map(platform_priority).fillna(0)
    aliases = aliases.sort_values(["season", "platform_priority", "team_name"])
    latest = aliases.groupby("owner_id").tail(1)
    for _, row in latest.iterrows():
        display_map[row["owner_id"]] = row["team_name"]
    return display_map

def label_owner(owner_id: str) -> str:
    """Return the latest franchise name for display."""
    return OWNER_DISPLAY_MAP.get(owner_id, owner_id)

def label_owner_series(series: pd.Series) -> pd.Series:
    """Vectorized owner label mapping."""
    return series.map(label_owner)


@st.cache_data
def get_filtered_matchups(filter_option: str) -> pd.DataFrame:
    """Load matchups filtered by season type."""
    matchups_df = load_matchups().copy()
    matchups_df["is_playoffs_effective"] = matchups_df["is_playoffs"]
    missing = matchups_df["is_playoffs_effective"].isna()
    matchups_df.loc[missing, "is_playoffs_effective"] = matchups_df.loc[missing, "week"] >= 15
    matchups_df["is_playoffs_effective"] = matchups_df["is_playoffs_effective"].astype(bool)

    if filter_option == "Regular Season":
        return matchups_df[matchups_df["is_playoffs_effective"] == False].copy()
    if filter_option == "Playoffs":
        return matchups_df[matchups_df["is_playoffs_effective"] == True].copy()
    return matchups_df


OWNER_DISPLAY_MAP = get_owner_display_map()


def build_owner_season_from_matchups(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Build owner-season stats from a matchup DataFrame."""
    season_stats = []
    for season in matchups_df["season"].unique():
        season_games = matchups_df[matchups_df["season"] == season]
        owners = set(season_games["owner_id_home"]) | set(season_games["owner_id_away"])

        for owner_id in owners:
            home_games = season_games[season_games["owner_id_home"] == owner_id]
            away_games = season_games[season_games["owner_id_away"] == owner_id]

            wins = len(home_games[home_games["winner_owner_id"] == owner_id]) + \
                   len(away_games[away_games["winner_owner_id"] == owner_id])
            losses = len(home_games[home_games["winner_owner_id"] == home_games["owner_id_away"]]) + \
                     len(away_games[away_games["winner_owner_id"] == away_games["owner_id_home"]])
            ties = len(home_games[home_games["winner_owner_id"] == "tie"]) + \
                   len(away_games[away_games["winner_owner_id"] == "tie"])

            points_for = home_games["score_home"].sum() + away_games["score_away"].sum()
            points_against = home_games["score_away"].sum() + away_games["score_home"].sum()

            games = wins + losses + ties
            win_pct = wins / games if games > 0 else 0.0

            season_stats.append({
                "season": season,
                "owner_id": owner_id,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_pct": round(win_pct, 4),
                "points_for": round(points_for, 2),
                "points_against": round(points_against, 2)
            })

    return pd.DataFrame(season_stats)


def build_owner_all_time_from_season(season_df: pd.DataFrame) -> pd.DataFrame:
    """Build all-time owner stats from owner-season stats."""
    all_time_stats = []
    for owner_id in season_df["owner_id"].unique():
        owner_seasons = season_df[season_df["owner_id"] == owner_id]

        wins = owner_seasons["wins"].sum()
        losses = owner_seasons["losses"].sum()
        ties = owner_seasons["ties"].sum()
        games = wins + losses + ties

        points_for = owner_seasons["points_for"].sum()
        points_against = owner_seasons["points_against"].sum()

        win_pct = wins / games if games > 0 else 0.0
        avg_points_for = points_for / games if games > 0 else 0.0
        avg_points_against = points_against / games if games > 0 else 0.0

        all_time_stats.append({
            "owner_id": owner_id,
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_pct": round(win_pct, 4),
            "points_for": round(points_for, 2),
            "points_against": round(points_against, 2),
            "avg_points_for": round(avg_points_for, 2),
            "avg_points_against": round(avg_points_against, 2)
        })

    return pd.DataFrame(all_time_stats).sort_values("win_pct", ascending=False).reset_index(drop=True)


def build_h2h_from_matchups(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Build head-to-head stats from a matchup DataFrame."""
    h2h_stats = []
    owners = set(matchups_df["owner_id_home"]) | set(matchups_df["owner_id_away"])

    for owner_a in owners:
        for owner_b in owners:
            if owner_a >= owner_b:
                continue

            matchups_ab = matchups_df[
                ((matchups_df["owner_id_home"] == owner_a) & (matchups_df["owner_id_away"] == owner_b)) |
                ((matchups_df["owner_id_home"] == owner_b) & (matchups_df["owner_id_away"] == owner_a))
            ]

            if matchups_ab.empty:
                continue

            a_wins = len(matchups_ab[matchups_ab["winner_owner_id"] == owner_a])
            b_wins = len(matchups_ab[matchups_ab["winner_owner_id"] == owner_b])
            ties = len(matchups_ab[matchups_ab["winner_owner_id"] == "tie"])
            games = len(matchups_ab)

            a_home = matchups_ab[matchups_ab["owner_id_home"] == owner_a]
            a_away = matchups_ab[matchups_ab["owner_id_away"] == owner_a]
            a_points = a_home["score_home"].sum() + a_away["score_away"].sum()

            b_home = matchups_ab[matchups_ab["owner_id_home"] == owner_b]
            b_away = matchups_ab[matchups_ab["owner_id_away"] == owner_b]
            b_points = b_home["score_home"].sum() + b_away["score_away"].sum()

            h2h_stats.append({
                "owner_a": owner_a,
                "owner_b": owner_b,
                "games": games,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "a_points_for": round(a_points, 2),
                "b_points_for": round(b_points, 2)
            })

    return pd.DataFrame(h2h_stats)


@st.cache_data
def get_filtered_owner_season_stats(filter_option: str) -> pd.DataFrame:
    """Owner-season stats for the selected game filter."""
    matchups_df = get_filtered_matchups(filter_option)
    return build_owner_season_from_matchups(matchups_df)


@st.cache_data
def get_filtered_all_time_stats(filter_option: str) -> pd.DataFrame:
    """All-time owner stats for the selected game filter."""
    season_df = get_filtered_owner_season_stats(filter_option)
    if season_df.empty:
        return pd.DataFrame(columns=[
            "owner_id",
            "games",
            "wins",
            "losses",
            "ties",
            "win_pct",
            "points_for",
            "points_against",
            "avg_points_for",
            "avg_points_against"
        ])
    return build_owner_all_time_from_season(season_df)


@st.cache_data
def get_filtered_h2h_stats(filter_option: str) -> pd.DataFrame:
    """Head-to-head stats for the selected game filter."""
    matchups_df = get_filtered_matchups(filter_option)
    if matchups_df.empty:
        return pd.DataFrame(columns=[
            "owner_a",
            "owner_b",
            "games",
            "a_wins",
            "b_wins",
            "ties",
            "a_points_for",
            "b_points_for"
        ])
    return build_h2h_from_matchups(matchups_df)


@st.cache_data
def get_league_benchmarks(filter_option: str) -> dict:
    """Calculate league-wide benchmarks for context."""
    matchups_df = get_filtered_matchups(filter_option)
    season_df = build_owner_season_from_matchups(matchups_df)
    all_time_df = build_owner_all_time_from_season(season_df)

    return {
        'avg_win_pct': all_time_df['win_pct'].mean() if not all_time_df.empty else 0.0,
        'avg_ppg': all_time_df['avg_points_for'].mean() if not all_time_df.empty else 0.0,
        'total_games': len(matchups_df) // 2,
        'seasons': matchups_df['season'].nunique(),
        'active_owners': len(all_time_df)
    }

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_record(wins: int, losses: int, ties: int) -> str:
    """Format W-L-T record consistently."""
    return f"{wins}-{losses}-{ties}"

def format_pct(value: float) -> str:
    """Format percentage with 1 decimal."""
    return f"{value*100:.1f}%"

def get_rank_suffix(rank: int) -> str:
    """Get ordinal suffix for rank (1st, 2nd, 3rd, etc.)."""
    if 10 <= rank % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')
    return f"{rank}{suffix}"

def create_bar_chart(data: pd.DataFrame, x: str, y: str, title: str,
                     color_col: str = None, benchmark: float = None) -> go.Figure:
    """
    Create a professional horizontal bar chart with Bloomberg styling.
    Follows The Athletic principle: direct labeling, no legends.
    """
    fig = go.Figure()

    # Convert to lists for Plotly compatibility
    x_values = data[x].tolist()
    y_values = data[y].tolist()

    # Format text labels
    if x in ['win_pct']:
        text_labels = [f"{v*100:.1f}%" for v in x_values]
    else:
        text_labels = [f"{v:.1f}" for v in x_values]

    # Add bars
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        orientation='h',
        marker=dict(
            color=COLORS['primary'],
            line=dict(color=COLORS['primary'], width=0)
        ),
        text=text_labels,
        textposition='outside',
        textfont=dict(size=12, color='#000000', family='Inter'),
        hovertemplate='<b>%{y}</b><br>%{x:.2f}<extra></extra>'
    ))

    # Add benchmark line if provided
    if benchmark is not None:
        fig.add_vline(
            x=benchmark,
            line_dash="dash",
            line_color=COLORS['accent'],
            annotation_text=f"League Avg: {benchmark:.3f}" if x == 'win_pct' else f"League Avg: {benchmark:.1f}",
            annotation_position="top",
            annotation_font=dict(size=12, color='#000000', family='Inter')
        )

    # Clean, minimal layout
    xaxis_config = dict(
        showgrid=True,
        gridcolor='#E5E5E7',
        gridwidth=1,
        zeroline=False,
        title=None,
        tickfont=dict(size=12, color='#000000', family='Inter')
    )

    # Set appropriate range for win_pct
    if x == 'win_pct':
        xaxis_config['range'] = [0, 1]  # 0-100% in decimal
        xaxis_config['tickformat'] = '.1%'  # Format as percentage

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#000000', family='Inter'),
            x=0.02
        ),
        xaxis=xaxis_config,
        yaxis=dict(
            showgrid=False,
            title=None,
            categoryorder='total ascending',
            tickfont=dict(size=12, color='#000000', family='Inter')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(400, len(data) * 30),
        margin=dict(l=150, r=50, t=80, b=50),
        font=dict(family='Inter', color='#000000', size=12),
        showlegend=False
    )

    return fig

def create_line_chart(data: pd.DataFrame, x: str, y_cols: list,
                      title: str, labels: dict = None) -> go.Figure:
    """
    Create a professional line chart for time series.
    Uses direct labeling on lines instead of legends.
    """
    fig = go.Figure()

    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['danger']]

    for idx, col in enumerate(y_cols):
        label = labels.get(col, col) if labels else col
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[col],
            mode='lines+markers',
            name=label,
            line=dict(color=colors_list[idx % len(colors_list)], width=2.5),
            marker=dict(size=6),
            hovertemplate=f'<b>{label}</b><br>Season: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#000000', family='Inter'),
            x=0.02
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E5E7',
            title=None,
            dtick=1,
            tickfont=dict(size=12, color='#000000', family='Inter')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E5E5E7',
            title=None,
            zeroline=False,
            tickfont=dict(size=12, color='#000000', family='Inter')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family='Inter', color='#000000', size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#000000', family='Inter')
        )
    )

    return fig

def render_data_attribution(filter_option: str):
    """
    Bloomberg standard: Always show data sources and timestamps.
    """
    data_dir = Path("data/mart")
    last_updated = None

    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            last_updated = max(f.stat().st_mtime for f in csv_files)
            last_updated = datetime.fromtimestamp(last_updated).strftime("%B %d, %Y at %I:%M %p")

    matchups_df = get_filtered_matchups(filter_option)
    total_games = len(matchups_df) // 2
    total_seasons = matchups_df['season'].nunique()

    st.markdown("---")
    st.markdown(
        f"""
        <div class="footer">
        <strong>Data Sources:</strong> MyFantasyLeague (2013-2019), Fleaflicker (2020-2024), Sleeper (2025)<br>
        <strong>Last Updated:</strong> {last_updated or 'Unknown'}<br>
        <strong>Total Games:</strong> {total_games:,} | <strong>Seasons:</strong> {total_seasons} | <strong>Methodology:</strong> {filter_option} view of all matchups
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.markdown("# 🏆 Cataneration Trophy Room")
st.sidebar.markdown("**12+ Years of Fantasy Football History**")
st.sidebar.markdown("*2013-2025 • 1,160 Games*")
st.sidebar.markdown("---")

game_filter = st.sidebar.selectbox(
    "Game Filter",
    ["All Games", "Regular Season", "Playoffs"],
    help="Apply filter across leaderboards and matchup views"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Trophy Room",
        "All-Time Leaderboard",
        "Owner Profile",
        "Head-to-Head Explorer",
        "Weekly Timeline"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Quick stats in sidebar
benchmarks = get_league_benchmarks(game_filter)
st.sidebar.markdown("### League Overview")
st.sidebar.caption(f"View: {game_filter}")
st.sidebar.metric("Total Games", f"{benchmarks['total_games']:,}")
st.sidebar.metric("Active Owners", benchmarks['active_owners'])
st.sidebar.metric("Seasons", benchmarks['seasons'])
st.sidebar.metric("Avg Win %", f"{benchmarks['avg_win_pct']*100:.1f}%")

# ==============================================================================
# PAGE 1: TROPHY ROOM
# ==============================================================================

if page == "Trophy Room":
    st.title("Trophy Room")
    st.caption("Champions, #1 Losers, and season-by-season hardware")

    titles_df = load_titles()
    seeds_df = load_playoff_seeds()

    if titles_df.empty or titles_df["champion_owner_id"].isna().all():
        st.info("Titles not available yet. Ingest playoff brackets and rebuild marts.")
    else:
        latest_season = titles_df["season"].max()
        latest_title = titles_df[titles_df["season"] == latest_season].iloc[0]
        champion_label = latest_title["champion_owner_id"]
        champion_team = latest_title["champion_team_name"]
        loser_label = latest_title["losers_champ_owner_id"]
        loser_team = latest_title["losers_champ_team_name"]
        champion_years = (
            titles_df.loc[titles_df["champion_owner_id"] == champion_label, "season"]
            .dropna()
            .astype(int)
            .sort_values()
            .tolist()
        )
        champion_years_label = ", ".join(str(year) for year in champion_years)
        banner_title = champion_team if isinstance(champion_team, str) else champion_label
        latest_loser_label = label_owner(loser_label) if pd.notna(loser_label) else None

        champions = (
            titles_df.dropna(subset=["champion_owner_id"])
            .groupby("champion_owner_id")
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
        )
        losers = (
            titles_df.dropna(subset=["losers_champ_owner_id"])
            .groupby("losers_champ_owner_id")
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
        )

        top_champ = champions.iloc[0]
        top_loser = losers.iloc[0] if not losers.empty else None
        champions_display = champions.copy()
        champions_display["owner_label"] = label_owner_series(champions_display["champion_owner_id"])
        losers_display = losers.copy()
        if not losers_display.empty:
            losers_display["owner_label"] = label_owner_series(losers_display["losers_champ_owner_id"])

        st.markdown(
            f"""
            <div class="trophy-hero">
              <div class="trophy-micro">Latest Crown • {latest_season}</div>
              <div class="trophy-title">{banner_title}</div>
              <div class="trophy-subtitle">{champion_years_label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if game_filter != "All Games":
            st.caption("Titles always reflect full seasons (playoffs + regular season).")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Titles", label_owner(top_champ["champion_owner_id"]), int(top_champ["titles"]))
        with col2:
            if top_loser is not None:
                st.metric("#1 Loser Titles", label_owner(top_loser["losers_champ_owner_id"]), int(top_loser["titles"]))
            else:
                st.metric("#1 Loser Titles", "None", 0)
        with col3:
            st.metric("Seasons Tracked", titles_df["season"].nunique())

        st.markdown("---")

        recent_window = titles_df[titles_df["season"] >= latest_season - 4]
        recent_unique = recent_window["champion_owner_id"].nunique()
        parity_score = round(recent_unique / max(1, len(recent_window)), 2)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Title Momentum (Top 10)")
            top_titles = champions.head(10)
            top_titles = champions_display.head(10)
            fig = create_bar_chart(top_titles, "titles", "owner_label", "Most Titles")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Parity Check")
            st.metric("Unique Champs (Last 5)", recent_unique)
            st.metric("Parity Index", f"{parity_score:.2f}")
            if pd.notna(loser_label):
                st.markdown("###")
                st.markdown("##### #1 Loser (Latest)")
                st.markdown(f"**{latest_loser_label}**")
                st.caption(loser_team if isinstance(loser_team, str) else "No team name")
            else:
                st.caption("No #1 Loser recorded for latest season.")

        st.subheader("Championship Leaders")
        st.dataframe(
            champions_display.rename(columns={
                "owner_label": "Owner",
                "titles": "Titles"
            })[["Owner", "Titles"]],
            hide_index=True,
            use_container_width=True,
            height=min(600, len(champions) * 40 + 100)
        )

        st.subheader("#1 Loser Leaders")
        if losers.empty:
            st.info("No #1 Loser titles found yet.")
        else:
            losers_table = losers_display.rename(columns={
                "owner_label": "Owner",
                "titles": "#1 Loser Titles"
            })[["Owner", "#1 Loser Titles"]]
            st.dataframe(
                losers_table,
                hide_index=True,
                use_container_width=True,
                height=min(600, len(losers) * 40 + 100)
            )

        st.subheader("Season Titles")
        display_titles = titles_df.copy()
        display_titles = display_titles.sort_values("season")
        display_titles = display_titles[[
            "season",
            "platform",
            "champion_owner_id",
            "champion_team_name",
            "losers_champ_owner_id",
            "losers_champ_team_name"
        ]]
        display_titles["champion_owner_id"] = label_owner_series(display_titles["champion_owner_id"])
        display_titles["losers_champ_owner_id"] = label_owner_series(display_titles["losers_champ_owner_id"])
        display_titles.columns = [
            "Season",
            "Platform",
            "Champion",
            "Champion Team",
            "#1 Loser",
            "#1 Loser Team"
        ]
        st.dataframe(
            display_titles,
            hide_index=True,
            use_container_width=True,
            height=min(600, len(display_titles) * 40 + 100)
        )

        st.subheader("Playoff Seeding by Season")
        if seeds_df.empty:
            st.info("Playoff seeding not available yet. Rebuild marts to generate it.")
        else:
            # Filter to winners bracket only and playoff teams (seeds 1-6)
            winners_seeds = seeds_df[seeds_df["bracket_type"] == "winners"].copy()
            if winners_seeds.empty:
                st.info("No playoff seeding data available.")
            else:
                # Get unique owner/seed per season, only playoff seeds (1-6)
                winners_seeds = winners_seeds.dropna(subset=["seed"])
                winners_seeds = winners_seeds[winners_seeds["seed"] <= 6]
                winners_seeds = winners_seeds.drop_duplicates(subset=["season", "owner_id", "seed"])
                winners_seeds["Owner"] = label_owner_series(winners_seeds["owner_id"])
                winners_seeds["Team"] = winners_seeds["team_name"].fillna("")
                winners_seeds["Seed"] = winners_seeds["seed"].astype(int)

                seed_seasons = sorted(winners_seeds["season"].dropna().unique(), reverse=True)
                seed_season = st.selectbox("Season", seed_seasons, key="seed_season")

                season_seeds = winners_seeds[winners_seeds["season"] == seed_season].copy()
                if season_seeds.empty:
                    st.info("No seeding data for this season.")
                else:
                    # Sort by seed and display
                    season_seeds = season_seeds.sort_values("Seed")
                    display_seeds = season_seeds[["Seed", "Owner", "Team"]].reset_index(drop=True)
                    st.dataframe(
                        display_seeds,
                        hide_index=True,
                        use_container_width=True,
                        height=min(400, len(display_seeds) * 36 + 80)
                    )

    render_data_attribution(game_filter)

# ==============================================================================
# PAGE 2: ALL-TIME LEADERBOARD
# ==============================================================================

elif page == "All-Time Leaderboard":
    st.title("All-Time Leaderboard")
    st.caption(f"Comprehensive performance rankings • {game_filter}")

    # Load data
    df = get_filtered_all_time_stats(game_filter)
    benchmarks = get_league_benchmarks(game_filter)
    titles_df = load_titles()

    if df.empty:
        st.info("No games found for this filter.")
        render_data_attribution(game_filter)
        st.stop()

    df_display = df.copy()
    df_display["owner_label"] = label_owner_series(df_display["owner_id"])

    if not titles_df.empty and titles_df["champion_owner_id"].notna().any():
        latest_season = titles_df["season"].max()
        latest_title = titles_df[titles_df["season"] == latest_season].iloc[0]
        champion_label = latest_title["champion_owner_id"]
        champion_team = latest_title["champion_team_name"]
        loser_label = latest_title["losers_champ_owner_id"]
        loser_team = latest_title["losers_champ_team_name"]
        champion_years = (
            titles_df.loc[titles_df["champion_owner_id"] == champion_label, "season"]
            .dropna()
            .astype(int)
            .sort_values()
            .tolist()
        )
        champion_years_label = ", ".join(str(year) for year in champion_years)
        banner_title = champion_team if isinstance(champion_team, str) else champion_label
        latest_loser_label = label_owner(loser_label) if pd.notna(loser_label) else None

        champions = (
            titles_df.dropna(subset=["champion_owner_id"])
            .groupby("champion_owner_id")
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
        )
        losers = (
            titles_df.dropna(subset=["losers_champ_owner_id"])
            .groupby("losers_champ_owner_id")
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
        )

        top_champ = champions.iloc[0]
        top_loser = losers.iloc[0] if not losers.empty else None
        champions_display = champions.copy()
        champions_display["owner_label"] = label_owner_series(champions_display["champion_owner_id"])
        losers_display = losers.copy()
        if not losers_display.empty:
            losers_display["owner_label"] = label_owner_series(losers_display["losers_champ_owner_id"])

        st.markdown(
            f"""
            <div class="trophy-hero">
              <div class="trophy-micro">Latest Crown • {latest_season}</div>
              <div class="trophy-title">{banner_title}</div>
              <div class="trophy-subtitle">{champion_years_label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Titles", label_owner(top_champ["champion_owner_id"]), int(top_champ["titles"]))
        with col2:
            if top_loser is not None:
                st.metric("#1 Loser Titles", label_owner(top_loser["losers_champ_owner_id"]), int(top_loser["titles"]))
            else:
                st.metric("#1 Loser Titles", "None", 0)
        with col3:
            st.metric("Seasons Tracked", titles_df["season"].nunique())

        st.markdown("---")

    # Key metrics at top (Bloomberg style: 4 key numbers)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_owner = df.iloc[0]
        st.metric(
            "League Leader",
            label_owner(top_owner['owner_id']),
            format_pct(top_owner['win_pct'])
        )
    with col2:
        best_scorer = df.loc[df['avg_points_for'].idxmax()]
        st.metric(
            "Highest Avg PPG",
            label_owner(best_scorer['owner_id']),
            f"{best_scorer['avg_points_for']:.1f}"
        )
    with col3:
        most_games = df.loc[df['games'].idxmax()]
        st.metric(
            "Most Games",
            label_owner(most_games['owner_id']),
            int(most_games['games'])
        )
    with col4:
        st.metric(
            "League Avg Win %",
            format_pct(benchmarks['avg_win_pct']),
            delta=None
        )

    st.markdown("###")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Rankings Table", "📈 Win % Chart", "🎯 Points Scoring Chart", "🍀 Luck & PF vs W/L"]
    )

    with tab1:
        # Prepare display dataframe
        display_df = df_display.copy()
        display_df['rank'] = range(1, len(display_df) + 1)
        display_df['record'] = display_df.apply(
            lambda x: format_record(x['wins'], x['losses'], x['ties']), axis=1
        )

        # Style the dataframe
        styled_df = display_df[[
            'rank', 'owner_label', 'games', 'record', 'win_pct',
            'points_for', 'points_against', 'avg_points_for', 'avg_points_against'
        ]].copy()

        styled_df['win_pct'] = styled_df['win_pct'] * 100

        styled_df.columns = [
            'Rank', 'Owner', 'GP', 'Record', 'Win %',
            'PF', 'PA', 'PPG', 'PAPG'
        ]

        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=min(800, len(styled_df) * 40 + 100),
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "GP": st.column_config.NumberColumn(format="%d"),
                "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "PF": st.column_config.NumberColumn(format="%.1f"),
                "PA": st.column_config.NumberColumn(format="%.1f"),
                "PPG": st.column_config.NumberColumn(format="%.2f"),
                "PAPG": st.column_config.NumberColumn(format="%.2f"),
            }
        )

        # Podium display (The Athletic style: visual storytelling)
        st.markdown("###")
        st.markdown("### 🏆 Podium")
        col1, col2, col3 = st.columns(3)

        for idx, col in enumerate([col1, col2, col3]):
            owner = df.iloc[idx]
            emoji = ['🥇', '🥈', '🥉'][idx]
            with col:
                st.markdown(f"### {emoji} {get_rank_suffix(idx+1)} Place")
                st.markdown(f"**{label_owner(owner['owner_id'])}**")
                st.caption(f"{format_record(owner['wins'], owner['losses'], owner['ties'])} ({format_pct(owner['win_pct'])})")
                st.caption(f"{owner['avg_points_for']:.2f} PPG • {int(owner['games'])} games")

    with tab2:
        # Win percentage bar chart with league average benchmark
        chart_df = df_display.head(15).copy()  # Top 15 for readability
        fig = create_bar_chart(
            chart_df,
            x='win_pct',
            y='owner_label',
            title='Win Percentage Leaders (Min. Games Played)',
            benchmark=benchmarks['avg_win_pct']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("*Chart shows top 15 owners by win percentage. Dashed line indicates league average.*")

    with tab3:
        # Points per game comparison
        chart_df = df_display.head(15).copy()
        fig = create_bar_chart(
            chart_df,
            x='avg_points_for',
            y='owner_label',
            title='Average Points Per Game Leaders',
            benchmark=benchmarks['avg_ppg']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("*Chart shows top 15 owners by average points per game. Dashed line indicates league average.*")

    with tab4:
        luck_df = df_display.copy()
        luck_df['expected_win_pct'] = (
            (luck_df['points_for'] ** 2) /
            ((luck_df['points_for'] ** 2) + (luck_df['points_against'] ** 2))
        )
        luck_df['luck_factor'] = luck_df['win_pct'] - luck_df['expected_win_pct']
        luck_df['luck_label'] = luck_df['luck_factor'].apply(
            lambda x: "▲ Lucky" if x > 0 else ("▼ Unlucky" if x < 0 else "Even")
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=luck_df['avg_points_for'].tolist(),
            y=(luck_df['win_pct'] * 100).tolist(),
            mode='markers',
            text=luck_df['owner_label'].tolist(),
            customdata=list(zip(
                luck_df['games'].tolist(),
                luck_df['luck_label'].tolist(),
                (luck_df['luck_factor'] * 100).tolist(),
                (luck_df['win_pct'] * 100).tolist(),
                luck_df['points_for'].tolist(),
                luck_df['points_against'].tolist()
            )),
            marker=dict(
                size=luck_df['games'].tolist(),
                sizemode='area',
                sizeref=2.0 * max(luck_df['games'].tolist()) / (40 ** 2),
                sizemin=6,
                color=luck_df['luck_factor'].tolist(),
                colorscale=[[0.0, '#FF3B30'], [0.5, '#F5F5F7'], [1.0, '#00B271']],
                colorbar=dict(title="Luck"),
                line=dict(width=0)
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PPG: %{x:.2f}<br>"
                "Win %: %{y:.1f}%<br>"
                "Games: %{customdata[0]}<br>"
                "Luck: %{customdata[1]} (%{customdata[2]:+.1f}%)<br>"
                "PF: %{customdata[4]:.1f}<br>"
                "PA: %{customdata[5]:.1f}"
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title=dict(
                text="Points For vs Win % (Luck View)",
                font=dict(size=18, color='#000000', family='Inter'),
                x=0.02
            ),
            xaxis=dict(
                title=dict(
                    text="Average Points Per Game",
                    font=dict(size=12, color='#000000', family='Inter')
                ),
                showgrid=True,
                gridcolor='#E5E5E7'
            ),
            yaxis=dict(
                title=dict(
                    text="Win %",
                    font=dict(size=12, color='#000000', family='Inter')
                ),
                showgrid=True,
                gridcolor='#E5E5E7'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            coloraxis_colorbar=dict(title="Luck"),
            font=dict(family='Inter', color='#000000', size=12)
        )

        fig.add_shape(
            type="line",
            x0=benchmarks['avg_ppg'],
            x1=benchmarks['avg_ppg'],
            y0=luck_df['win_pct'].min() * 100,
            y1=luck_df['win_pct'].max() * 100,
            line=dict(color="#999999", width=1, dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=luck_df['avg_points_for'].min(),
            x1=luck_df['avg_points_for'].max(),
            y0=benchmarks['avg_win_pct'] * 100,
            y1=benchmarks['avg_win_pct'] * 100,
            line=dict(color="#999999", width=1, dash="dash")
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("*Red indicates underperforming W/L vs points. Green indicates overperforming.*")

    # Export functionality
    st.markdown("###")
    csv = styled_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Leaderboard (CSV)",
        data=csv,
        file_name=f"cataneration_all_time_leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    render_data_attribution(game_filter)

# ==============================================================================
# PAGE 2: OWNER PROFILE
# ==============================================================================

elif page == "Owner Profile":
    st.title("Owner Profile")
    st.caption(f"Deep dive into individual season-by-season performance • {game_filter}")

    # Load data
    season_df = get_filtered_owner_season_stats(game_filter)
    all_time_df = get_filtered_all_time_stats(game_filter)
    achievements_df = load_owner_achievements()
    benchmarks = get_league_benchmarks(game_filter)

    if all_time_df.empty:
        st.info("No games found for this filter.")
        render_data_attribution(game_filter)
        st.stop()

    # Owner selector with search
    owners = sorted(all_time_df['owner_id'].unique(), key=lambda oid: label_owner(oid).lower())
    selected_owner = st.selectbox(
        "Select Owner",
        owners,
        format_func=label_owner,
        help="Choose an owner to view their complete performance history"
    )

    if selected_owner:
        owner_label = label_owner(selected_owner)
        # Get all-time stats for this owner
        owner_stats = all_time_df[all_time_df['owner_id'] == selected_owner].iloc[0]
        owner_rank = all_time_df[all_time_df['owner_id'] == selected_owner].index[0] + 1

        # Header with owner name and rank
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {owner_label}")
        with col2:
            st.markdown(f"### {get_rank_suffix(owner_rank)} Overall")

        st.markdown("###")

        # Key metrics (Bloomberg: 4-6 key numbers)
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                "All-Time Record",
                format_record(owner_stats['wins'], owner_stats['losses'], owner_stats['ties'])
            )

        with col2:
            delta_vs_avg = (owner_stats['win_pct'] - benchmarks['avg_win_pct']) * 100
            st.metric(
                "Win %",
                format_pct(owner_stats['win_pct']),
                f"{delta_vs_avg:+.1f}% vs avg"
            )

        with col3:
            st.metric(
                "Games Played",
                int(owner_stats['games'])
            )

        with col4:
            st.metric(
                "Total PF",
                f"{owner_stats['points_for']:,.1f}"
            )

        with col5:
            ppg_delta = owner_stats['avg_points_for'] - benchmarks['avg_ppg']
            st.metric(
                "PPG",
                f"{owner_stats['avg_points_for']:.2f}",
                f"{ppg_delta:+.2f} vs avg"
            )

        with col6:
            point_diff = owner_stats['points_for'] - owner_stats['points_against']
            st.metric(
                "Point Differential",
                f"{point_diff:+.1f}",
                "Total career"
            )

        st.markdown("---")

        # Get season data for this owner (needed by all tabs)
        owner_seasons = season_df[season_df['owner_id'] == selected_owner].sort_values('season')

        # Season-by-season tabs
        tab1, tab2, tab3 = st.tabs(["📅 Season History", "📈 Trend Analysis", "🎯 Performance Metrics"])

        with tab1:
            if len(owner_seasons) > 0:
                st.subheader("Season-by-Season Breakdown")

                display_seasons = owner_seasons.copy()
                display_seasons['record'] = display_seasons.apply(
                    lambda x: format_record(x['wins'], x['losses'], x['ties']), axis=1
                )
                display_seasons['win_pct_pct'] = display_seasons['win_pct'] * 100

                # Calculate vs league average for each season
                league_avg_by_season = season_df.groupby('season')['win_pct'].mean()
                display_seasons['vs_avg_pct'] = display_seasons.apply(
                    lambda x: (x['win_pct'] - league_avg_by_season.get(x['season'], 0.5)) * 100,
                    axis=1
                )

                styled_seasons = display_seasons[[
                    'season', 'record', 'win_pct_pct', 'vs_avg_pct',
                    'points_for', 'points_against'
                ]].copy()

                styled_seasons.columns = [
                    'Season', 'Record', 'Win %', 'vs League Avg', 'PF', 'PA'
                ]

                st.dataframe(
                    styled_seasons,
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, len(styled_seasons) * 40 + 100),
                    column_config={
                        "Season": st.column_config.NumberColumn(format="%d"),
                        "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                        "vs League Avg": st.column_config.NumberColumn(format="%+.1f%%"),
                        "PF": st.column_config.NumberColumn(format="%.1f"),
                        "PA": st.column_config.NumberColumn(format="%.1f"),
                    }
                )

                # Summary stats
                st.markdown("###")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Season",
                             f"{owner_seasons.loc[owner_seasons['win_pct'].idxmax(), 'season']:.0f}",
                             format_pct(owner_seasons['win_pct'].max()))
                with col2:
                    st.metric("Worst Season",
                             f"{owner_seasons.loc[owner_seasons['win_pct'].idxmin(), 'season']:.0f}",
                             format_pct(owner_seasons['win_pct'].min()))
                with col3:
                    st.metric("Best Scoring Season",
                             f"{owner_seasons.loc[owner_seasons['points_for'].idxmax(), 'season']:.0f}",
                             f"{owner_seasons['points_for'].max():.1f} PF")
                with col4:
                    seasons_above_500 = len(owner_seasons[owner_seasons['win_pct'] > 0.5])
                    st.metric("Winning Seasons",
                             f"{seasons_above_500}/{len(owner_seasons)}",
                             f"{seasons_above_500/len(owner_seasons)*100:.0f}%")
            else:
                st.info(f"No season data found for {owner_label}")

        with tab2:
            if len(owner_seasons) > 0:
                st.subheader("Performance Trends Over Time")

                # Data info removed for cleaner UI
                # Uncomment below for debugging:
                # st.write(f"📊 Data loaded: {len(owner_seasons)} seasons")
                # st.write(f"📅 Seasons: {owner_seasons['season'].min()} - {owner_seasons['season'].max()}")

                # Win percentage trend - convert to percentage for display
                fig = go.Figure()

                # Convert to lists for Plotly compatibility
                seasons_list = owner_seasons['season'].tolist()
                win_pct_list = (owner_seasons['win_pct'] * 100).tolist()

                # Add owner win percentage
                fig.add_trace(go.Scatter(
                    x=seasons_list,
                    y=win_pct_list,
                    mode='lines+markers',
                    name=owner_label,
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{owner_label}</b><br>Season: %{{x}}<br>Win %: %{{y:.1f}}%<extra></extra>'
                ))

                # Add league average line for context
                league_avg_by_season_df = season_df.groupby('season')['win_pct'].mean().reset_index()
                league_avg_by_season_df.columns = ['season', 'league_avg']

                # Merge to get league avg for each season this owner played
                trend_data = owner_seasons.merge(league_avg_by_season_df, on='season', how='left')

                # Convert league average to lists
                league_seasons = trend_data['season'].tolist()
                league_avg_list = (trend_data['league_avg'] * 100).tolist()

                fig.add_trace(go.Scatter(
                    x=league_seasons,
                    y=league_avg_list,
                    mode='lines',
                    name='League Average',
                    line=dict(color=COLORS['accent'], width=2, dash='dash'),
                    hovertemplate='<b>League Avg</b><br>Season: %{x}<br>Win %: %{y:.1f}%<extra></extra>'
                ))

                # Update layout for win percentage chart
                fig.update_layout(
                    title=dict(
                        text='Win Percentage by Season',
                        font=dict(size=18, color='#000000', family='Inter'),
                        x=0.02
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#E5E5E7',
                        title=None,
                        dtick=1,
                        tickfont=dict(size=12, color='#000000', family='Inter')
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#E5E5E7',
                        title='Win Percentage (%)',
                        zeroline=False,
                        tickfont=dict(size=12, color='#000000', family='Inter'),
                        ticksuffix='%',
                        range=[0, 100]
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(l=50, r=50, t=80, b=50),
                    font=dict(family='Inter', color='#000000', size=12),
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12, color='#000000', family='Inter')
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Points scoring trend
                fig2 = go.Figure()

                # Convert to lists for Plotly compatibility
                points_for_list = owner_seasons['points_for'].tolist()
                points_against_list = owner_seasons['points_against'].tolist()

                fig2.add_trace(go.Scatter(
                    x=seasons_list,  # Reuse from above
                    y=points_for_list,
                    mode='lines+markers',
                    name='Points For',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Points For</b><br>Season: %{x}<br>Points: %{y:.1f}<extra></extra>'
                ))

                fig2.add_trace(go.Scatter(
                    x=seasons_list,  # Reuse from above
                    y=points_against_list,
                    mode='lines+markers',
                    name='Points Against',
                    line=dict(color=COLORS['secondary'], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Points Against</b><br>Season: %{x}<br>Points: %{y:.1f}<extra></extra>'
                ))

                # Update layout for points chart
                fig2.update_layout(
                    title=dict(
                        text='Points For vs Points Against by Season',
                        font=dict(size=18, color='#000000', family='Inter'),
                        x=0.02
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#E5E5E7',
                        title=None,
                        dtick=1,
                        tickfont=dict(size=12, color='#000000', family='Inter')
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#E5E5E7',
                        title='Total Points',
                        zeroline=False,
                        tickfont=dict(size=12, color='#000000', family='Inter')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    margin=dict(l=50, r=50, t=80, b=50),
                    font=dict(family='Inter', color='#000000', size=12),
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12, color='#000000', family='Inter')
                    )
                )

                st.plotly_chart(fig2, use_container_width=True)

                st.caption("*Line charts show performance trends across all seasons. Dashed line represents league average.*")

        with tab3:
            if len(owner_seasons) > 0:
                if game_filter != "All Games":
                    st.info("Achievements reflect full seasons (all games), not the current filter.")

                if not achievements_df.empty and selected_owner in achievements_df["owner_id"].values:
                    owner_achievements = achievements_df[
                        achievements_df["owner_id"] == selected_owner
                    ].iloc[0]

                    st.subheader("Achievements")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Top-3 Winners Bracket",
                            int(owner_achievements.get("top3_winners_finishes", 0))
                        )
                        seasons = str(owner_achievements.get("top3_winners_seasons", "")).strip()
                        st.caption(f"Seasons: {seasons}" if seasons else "Seasons: None")

                    with col2:
                        st.metric(
                            "Top-3 PF Seasons",
                            int(owner_achievements["top3_pf_seasons"])
                        )
                        seasons = str(owner_achievements["top3_pf_seasons_list"]).strip()
                        st.caption(f"Seasons: {seasons}" if seasons else "Seasons: None")

                    with col3:
                        st.metric(
                            "Top-2 Playoff Seeds",
                            int(owner_achievements.get("top2_seeds", 0))
                        )
                        seasons = str(owner_achievements.get("top2_seeds_seasons", "")).strip()
                        st.caption(f"Seasons: {seasons}" if seasons else "Seasons: None")

                    st.markdown("---")
                else:
                    st.info("Achievements not available yet. Build marts to generate them.")

                st.subheader("Advanced Performance Metrics")

                # Calculate additional metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Consistency Metrics")
                    win_pct_std = owner_seasons['win_pct'].std()
                    ppg_std = owner_seasons['points_for'].std() / owner_seasons['wins'].add(owner_seasons['losses']).mean()

                    st.metric("Win % Std Dev", f"{win_pct_std:.3f}",
                             "Lower is more consistent")
                    st.metric("PPG Std Dev", f"{ppg_std:.2f}",
                             "Lower is more consistent")

                    # Streak analysis
                    st.markdown("###")
                    st.markdown("##### Season Quality")
                    great_seasons = len(owner_seasons[owner_seasons['win_pct'] >= 0.6])
                    good_seasons = len(owner_seasons[(owner_seasons['win_pct'] >= 0.5) & (owner_seasons['win_pct'] < 0.6)])
                    poor_seasons = len(owner_seasons[owner_seasons['win_pct'] < 0.5])

                    st.metric("Great Seasons (60%+)", great_seasons)
                    st.metric("Good Seasons (50-60%)", good_seasons)
                    st.metric("Below .500 Seasons", poor_seasons)

                with col2:
                    st.markdown("##### Scoring Efficiency")

                    # Points per win/loss
                    ppw = owner_stats['points_for'] / owner_stats['wins'] if owner_stats['wins'] > 0 else 0
                    ppl = owner_stats['points_for'] / owner_stats['losses'] if owner_stats['losses'] > 0 else 0

                    st.metric("Points Per Win", f"{ppw:.1f}")
                    st.metric("Points Per Loss", f"{ppl:.1f}")

                    # Pythagorean expectation (points-based expected wins)
                    pf = owner_stats['points_for']
                    pa = owner_stats['points_against']
                    expected_win_pct = (pf ** 2) / (pf ** 2 + pa ** 2)
                    luck_factor = owner_stats['win_pct'] - expected_win_pct

                    st.markdown("###")
                    st.markdown("##### Expected Win % (Pythagorean)")
                    st.metric("Expected Win %", format_pct(expected_win_pct))
                    st.metric(
                        "Win % vs Expected",
                        format_pct(owner_stats['win_pct']),
                        f"{luck_factor*100:+.1f}% vs expected"
                    )
                    st.caption("*Based on points scored vs allowed. Positive = won more than expected.*")

        # Export functionality
        st.markdown("###")
        if len(owner_seasons) > 0:
            csv = styled_seasons.to_csv(index=False)
            st.download_button(
                            label=f"📥 Export {owner_label} History (CSV)",
                data=csv,
                file_name=f"cataneration_{selected_owner}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    render_data_attribution(game_filter)

# ==============================================================================
# PAGE 3: HEAD-TO-HEAD EXPLORER
# ==============================================================================

elif page == "Head-to-Head Explorer":
    st.title("Head-to-Head Explorer")
    st.caption(f"Complete matchup history between any two owners • {game_filter}")

    # Load data
    h2h_df = get_filtered_h2h_stats(game_filter)
    all_time_df = get_filtered_all_time_stats(game_filter)
    matchups_df = get_filtered_matchups(game_filter)

    if all_time_df.empty or matchups_df.empty:
        st.info("No games found for this filter.")
        render_data_attribution(game_filter)
        st.stop()

    # Owner selectors
    owners = sorted(all_time_df['owner_id'].unique(), key=lambda oid: label_owner(oid).lower())

    col1, col2 = st.columns(2)
    with col1:
        owner_a = st.selectbox("Owner A", owners, key="owner_a", format_func=label_owner)
    with col2:
        available_opponents = [o for o in owners if o != owner_a]
        owner_b = st.selectbox("Owner B", available_opponents, key="owner_b", format_func=label_owner)

    if owner_a and owner_b:
        owner_a_label = label_owner(owner_a)
        owner_b_label = label_owner(owner_b)
        # Find H2H record
        owner_1, owner_2 = sorted([owner_a, owner_b])
        h2h_record = h2h_df[
            (h2h_df['owner_a'] == owner_1) & (h2h_df['owner_b'] == owner_2)
        ]

        if len(h2h_record) > 0:
            record = h2h_record.iloc[0]

            # Determine stats for each owner
            if owner_a == owner_1:
                a_wins = int(record['a_wins'])
                b_wins = int(record['b_wins'])
                a_pf = record['a_points_for']
                b_pf = record['b_points_for']
            else:
                a_wins = int(record['b_wins'])
                b_wins = int(record['a_wins'])
                a_pf = record['b_points_for']
                b_pf = record['a_points_for']

            ties = int(record['ties'])
            total_games = int(record['games'])

            # Header
            st.markdown(f"## {owner_a_label} vs {owner_b_label}")
            st.caption(f"All-time series: {total_games} games")
            st.markdown("###")

            # Win-Loss visualization
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"### {owner_a_label}")
                st.markdown(f"# {a_wins}")
                st.caption(f"**Wins** ({a_wins/total_games*100:.1f}%)")
                st.metric("Total Points", f"{a_pf:,.1f}")
                st.metric("PPG vs Opponent", f"{a_pf/total_games:.2f}")

            with col2:
                st.markdown("###")
                st.markdown(f"### vs")
                if ties > 0:
                    st.caption(f"{ties} ties")

            with col3:
                st.markdown(f"### {owner_b_label}")
                st.markdown(f"# {b_wins}")
                st.caption(f"**Wins** ({b_wins/total_games*100:.1f}%)")
                st.metric("Total Points", f"{b_pf:,.1f}")
                st.metric("PPG vs Opponent", f"{b_pf/total_games:.2f}")

            st.markdown("---")

            # Get game-by-game history
            games = matchups_df[
                ((matchups_df['owner_id_home'] == owner_a) & (matchups_df['owner_id_away'] == owner_b)) |
                ((matchups_df['owner_id_home'] == owner_b) & (matchups_df['owner_id_away'] == owner_a))
            ].sort_values(['season', 'week'])

            if len(games) > 0:
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📅 Game Log", "📊 Series Breakdown", "📈 Scoring Trends"])

                with tab1:
                    st.subheader("Complete Game History")

                    display_games = games.copy().sort_values(['season', 'week'])
                    display_games = display_games.reset_index(drop=True)

                    def format_picker_label(row):
                        if row['owner_id_home'] == owner_a:
                            home = owner_a_label
                            away = owner_b_label
                            home_score = row['score_home']
                            away_score = row['score_away']
                        else:
                            home = owner_b_label
                            away = owner_a_label
                            home_score = row['score_home']
                            away_score = row['score_away']

                        playoff_flag = row['is_playoffs_effective'] if 'is_playoffs_effective' in row else row['is_playoffs']
                        playoff_tag = " (Playoffs)" if playoff_flag else ""
                        platform_tag = row['platform'].upper()
                        return (
                            f"Game {row['game_num']} — {row['season']} Wk {row['week']} "
                            f"({platform_tag}): {home} {home_score:.2f} vs {away} {away_score:.2f}{playoff_tag}"
                        )

                    display_games['game_num'] = display_games.index + 1
                    display_games['game_label'] = display_games.apply(format_picker_label, axis=1)

                    st.markdown("##### Matchup Picker")
                    selected_label = st.selectbox(
                        "Matchup",
                        display_games['game_label'],
                        key="h2h_matchup_picker"
                    )
                    selected_game = display_games[display_games['game_label'] == selected_label].iloc[0]
                    margin = abs(selected_game['score_home'] - selected_game['score_away'])
                    winner = (
                        selected_game['owner_id_home']
                        if selected_game['score_home'] > selected_game['score_away']
                        else (
                            selected_game['owner_id_away']
                            if selected_game['score_away'] > selected_game['score_home']
                            else "TIE"
                        )
                    )
                    winner_label = label_owner(winner) if winner != "TIE" else "TIE"
                    st.caption(
                        f"Season {selected_game['season']} · Week {selected_game['week']} · "
                        f"{selected_game['platform'].upper()} · Winner: {winner_label} · Margin: {margin:.2f}"
                    )

                    # Determine winner and loser for each game
                    def format_matchup(row):
                        if row['owner_id_home'] == owner_a:
                            home = owner_a_label
                            away = owner_b_label
                            home_score = row['score_home']
                            away_score = row['score_away']
                        else:
                            home = owner_b_label
                            away = owner_a_label
                            home_score = row['score_home']
                            away_score = row['score_away']

                        # Bold the winner
                        if home_score > away_score:
                            return f"**{home}** {home_score:.2f} - {away_score:.2f} {away}"
                        elif away_score > home_score:
                            return f"{home} {home_score:.2f} - {away_score:.2f} **{away}**"
                        else:
                            return f"{home} {home_score:.2f} - {away_score:.2f} {away} (TIE)"

                    display_games['matchup'] = display_games.apply(format_matchup, axis=1)
                    if 'is_playoffs_effective' in display_games.columns:
                        display_games['playoffs'] = display_games['is_playoffs_effective'].apply(lambda x: '🏆' if x else '')
                    else:
                        display_games['playoffs'] = display_games['is_playoffs'].apply(lambda x: '🏆' if x else '')
                    display_games['winner'] = display_games.apply(
                        lambda x: x['owner_id_home'] if x['score_home'] > x['score_away']
                                 else (x['owner_id_away'] if x['score_away'] > x['score_home'] else 'TIE'),
                        axis=1
                    )
                    display_games['winner'] = label_owner_series(display_games['winner'])

                    styled_games = display_games[[
                        'season', 'week', 'matchup', 'winner', 'playoffs', 'platform'
                    ]].copy()
                    styled_games.columns = ['Season', 'Week', 'Matchup', 'Winner', 'Playoffs', 'Platform']

                    st.dataframe(
                        styled_games,
                        hide_index=True,
                        use_container_width=True,
                        height=min(600, len(styled_games) * 40 + 100)
                    )

                    # Export game log
                    csv = styled_games.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Export Game Log (CSV)",
                        data=csv,
                        file_name=f"cataneration_h2h_{owner_a}_vs_{owner_b}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                with tab2:
                    st.subheader("Series Statistics")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### Wins by Season")
                        wins_by_season_a = games[games['winner_owner_id'] == owner_a].groupby('season').size()
                        wins_by_season_b = games[games['winner_owner_id'] == owner_b].groupby('season').size()

                        seasons_played = sorted(games['season'].unique())
                        for season in seasons_played:
                            a_season_wins = wins_by_season_a.get(season, 0)
                            b_season_wins = wins_by_season_b.get(season, 0)
                            st.caption(f"**{season}:** {owner_a_label} {a_season_wins}-{b_season_wins} {owner_b_label}")

                    with col2:
                        st.markdown("##### Context")

                        games['is_playoffs_effective'] = games['is_playoffs']
                        missing_playoffs = games['is_playoffs_effective'].isna()
                        games.loc[missing_playoffs, 'is_playoffs_effective'] = games.loc[
                            missing_playoffs, 'week'
                        ] >= 15
                        playoff_games_df = games[games['is_playoffs_effective'] == True]
                        playoff_games = len(playoff_games_df)
                        regular_games = len(games[games['is_playoffs_effective'] == False])
                        playoff_wins_a = len(playoff_games_df[playoff_games_df['winner_owner_id'] == owner_a])
                        playoff_wins_b = len(playoff_games_df[playoff_games_df['winner_owner_id'] == owner_b])
                        playoff_ties = len(playoff_games_df[playoff_games_df['winner_owner_id'] == "tie"])
                        playoff_record = f"{owner_a_label} {playoff_wins_a}-{playoff_wins_b} {owner_b_label}"
                        if playoff_ties > 0:
                            playoff_record = f"{playoff_record} ({playoff_ties} ties)"

                        def build_matchup_list(df):
                            return [
                                f"{int(row.season)} Wk {int(row.week)}: "
                                f"{label_owner(row.owner_id_home)} {row.score_home:.2f} - "
                                f"{row.score_away:.2f} {label_owner(row.owner_id_away)}"
                                for row in df.itertuples(index=False)
                            ]

                        def render_matchup_list(label, items):
                            if items:
                                st.caption(label)
                                st.markdown("\n".join([f"- {item}" for item in items]))
                            else:
                                st.caption(f"{label} None")

                        playoff_matchups_list = build_matchup_list(playoff_games_df)
                        regular_games_df = games[games['is_playoffs_effective'] == False]
                        regular_matchups_list = build_matchup_list(regular_games_df)

                        st.markdown(f"**Playoff Matchups:** {playoff_games}")
                        st.markdown(f"**Playoff Record:** {playoff_record}")
                        render_matchup_list("Matchups:", playoff_matchups_list)
                        st.markdown(f"**Regular Season:** {regular_games}")
                        render_matchup_list("Matchups:", regular_matchups_list)

                        # Blowouts/close games
                        games['margin'] = abs(games['score_home'] - games['score_away'])
                        blowouts = len(games[games['margin'] >= BLOWOUT_MARGIN])
                        close_games = len(games[games['margin'] <= CLOSE_MARGIN])

                        st.markdown(f"**Blowouts ({BLOWOUT_MARGIN}+ pts):** {blowouts}")
                        blowout_games = games[games['margin'] >= BLOWOUT_MARGIN]
                        blowout_matchups_list = build_matchup_list(blowout_games)
                        render_matchup_list("Matchups:", blowout_matchups_list)
                        st.markdown(f"**Close Games (≤{CLOSE_MARGIN} pts):** {close_games}")
                        close_games_df = games[games['margin'] <= CLOSE_MARGIN]
                        close_matchups_list = build_matchup_list(close_games_df)
                        render_matchup_list("Matchups:", close_matchups_list)

                with tab3:
                    st.subheader("Scoring Over Time")

                    # Prepare data for visualization
                    viz_data = games.copy()

                    # Create columns for each owner's score
                    viz_data['owner_a_score'] = viz_data.apply(
                        lambda x: x['score_home'] if x['owner_id_home'] == owner_a else x['score_away'],
                        axis=1
                    )
                    viz_data['owner_b_score'] = viz_data.apply(
                        lambda x: x['score_home'] if x['owner_id_home'] == owner_b else x['score_away'],
                        axis=1
                    )

                    # Create game number
                    viz_data['game_num'] = range(1, len(viz_data) + 1)

                    # Convert to lists for Plotly compatibility
                    game_nums = viz_data['game_num'].tolist()
                    owner_a_scores = viz_data['owner_a_score'].tolist()
                    owner_b_scores = viz_data['owner_b_score'].tolist()

                    # Line chart of scores over time
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=game_nums,
                        y=owner_a_scores,
                        mode='lines+markers',
                        name=owner_a_label,
                        line=dict(color=COLORS['primary'], width=2.5),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{owner_a_label}</b><br>Game %{{x}}<br>Score: %{{y:.2f}}<extra></extra>'
                    ))

                    fig.add_trace(go.Scatter(
                        x=game_nums,
                        y=owner_b_scores,
                        mode='lines+markers',
                        name=owner_b_label,
                        line=dict(color=COLORS['secondary'], width=2.5),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{owner_b_label}</b><br>Game %{{x}}<br>Score: %{{y:.2f}}<extra></extra>'
                    ))

                    fig.update_layout(
                        title=dict(
                            text="Score by Game Number",
                            font=dict(size=18, color='#000000', family='Inter'),
                            x=0.02
                        ),
                        xaxis=dict(
                            title=dict(
                                text="Game Number",
                                font=dict(size=12, color='#000000', family='Inter')
                            ),
                            showgrid=True,
                            gridcolor='#E5E5E7',
                            tickfont=dict(size=12, color='#000000', family='Inter')
                        ),
                        yaxis=dict(
                            title=dict(
                                text="Points",
                                font=dict(size=12, color='#000000', family='Inter')
                            ),
                            showgrid=True,
                            gridcolor='#E5E5E7',
                            tickfont=dict(size=12, color='#000000', family='Inter')
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400,
                        hovermode='x unified',
                        font=dict(family='Inter', color='#000000', size=12),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=12, color='#000000', family='Inter')
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(f"*Each point represents a matchup. Hover for details.*")
            else:
                st.info("No games found between these owners")
        else:
            st.info(f"No head-to-head record found between {owner_a_label} and {owner_b_label}")

    render_data_attribution(game_filter)

# ==============================================================================
# PAGE 4: WEEKLY TIMELINE
# ==============================================================================

elif page == "Weekly Timeline":
    st.title("Weekly Timeline")
    st.caption(f"Explore every week of every season • {game_filter}")

    # Load data
    matchups_df = get_filtered_matchups(game_filter)

    if matchups_df.empty:
        st.info("No games found for this filter.")
        render_data_attribution(game_filter)
        st.stop()

    # Season and week selectors
    seasons = sorted(matchups_df['season'].unique(), reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        selected_season = st.selectbox("Season", seasons)

    # Filter weeks by selected season
    season_matchups = matchups_df[matchups_df['season'] == selected_season]
    weeks = sorted(season_matchups['week'].unique())

    with col2:
        selected_week = st.selectbox("Week", weeks)

    if selected_season and selected_week:
        # Get matchups for this week
        week_matchups = season_matchups[season_matchups['week'] == selected_week].copy()

        # Header
        platform = week_matchups.iloc[0]['platform'].upper()
        is_playoffs = week_matchups.iloc[0]['is_playoffs_effective']

        st.markdown(f"## {selected_season} Week {selected_week}")
        st.caption(f"Platform: {platform}")

        if is_playoffs:
            st.warning("🏆 **PLAYOFF WEEK**")

        st.markdown("###")

        # Week summary stats
        total_points = week_matchups['score_home'].sum() + week_matchups['score_away'].sum()
        avg_score = total_points / (len(week_matchups) * 2)
        highest_score = max(week_matchups['score_home'].max(), week_matchups['score_away'].max())
        lowest_score = min(week_matchups['score_home'].min(), week_matchups['score_away'].min())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matchups", len(week_matchups))
        with col2:
            st.metric("Avg Score", f"{avg_score:.2f}")
        with col3:
            st.metric("High Score", f"{highest_score:.2f}")
        with col4:
            st.metric("Low Score", f"{lowest_score:.2f}")

        st.markdown("---")

        # Cumulative wins chart (season-to-date)
        owners = sorted(
            set(season_matchups['owner_id_home']) | set(season_matchups['owner_id_away']),
            key=lambda oid: label_owner(oid).lower()
        )
        cumulative_rows = []
        cumulative_wins = {owner: 0.0 for owner in owners}

        for week in weeks:
            week_games = season_matchups[season_matchups['week'] == week]
            weekly_wins = {owner: 0.0 for owner in owners}

            for _, game in week_games.iterrows():
                if game['score_home'] > game['score_away']:
                    weekly_wins[game['owner_id_home']] += 1.0
                elif game['score_away'] > game['score_home']:
                    weekly_wins[game['owner_id_away']] += 1.0
                else:
                    weekly_wins[game['owner_id_home']] += 0.5
                    weekly_wins[game['owner_id_away']] += 0.5

            for owner in owners:
                cumulative_wins[owner] += weekly_wins.get(owner, 0.0)
                cumulative_rows.append({
                    "week": week,
                    "owner": owner,
                    "cumulative_wins": cumulative_wins[owner]
                })

        cumulative_df = pd.DataFrame(cumulative_rows)

        st.subheader("Cumulative Wins")
        st.caption("Season-to-date wins by week (ties = 0.5).")

        wins_fig = go.Figure()
        for owner in owners:
            owner_df = cumulative_df[cumulative_df['owner'] == owner]
            owner_label = label_owner(owner)
            wins_fig.add_trace(go.Scatter(
                x=owner_df['week'].tolist(),
                y=owner_df['cumulative_wins'].tolist(),
                mode='lines+markers',
                name=owner_label,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{owner_label}</b><br>"
                    "Week %{x}<br>"
                    "Wins %{y:.1f}<extra></extra>"
                )
            ))

        wins_fig.update_layout(
            xaxis_title="Week",
            yaxis_title="Cumulative Wins",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=420,
            hovermode="x unified",
            font=dict(family='Inter', color='#000000', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        wins_fig.update_xaxes(showgrid=True, gridcolor="#E5E5E7")
        wins_fig.update_yaxes(showgrid=True, gridcolor="#E5E5E7")

        st.plotly_chart(wins_fig, use_container_width=True)

        st.markdown("---")

        # Display matchups in a clean card format
        st.subheader("Matchups")

        # Sort by score differential for drama
        week_matchups['margin'] = abs(week_matchups['score_home'] - week_matchups['score_away'])
        week_matchups = week_matchups.sort_values('margin', ascending=False)

        for idx, game in week_matchups.iterrows():
            home_score = game['score_home']
            away_score = game['score_away']
            margin = abs(home_score - away_score)

            # Determine winner
            if home_score > away_score:
                winner = game['owner_id_home']
                loser = game['owner_id_away']
                winner_score = home_score
                loser_score = away_score
                home_emoji = "🏆"
                away_emoji = ""
            elif away_score > home_score:
                winner = game['owner_id_away']
                loser = game['owner_id_home']
                winner_score = away_score
                loser_score = home_score
                home_emoji = ""
                away_emoji = "🏆"
            else:
                winner = None
                loser = None
                winner_score = home_score
                loser_score = away_score
                home_emoji = "🤝"
                away_emoji = "🤝"

            # Create matchup card
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 3])
                home_label = label_owner(game['owner_id_home'])
                away_label = label_owner(game['owner_id_away'])

                with col1:
                    if home_score > away_score:
                        st.markdown(f"### {home_emoji} **{home_label}**")
                    else:
                        st.markdown(f"### {home_emoji} {home_label}")

                with col2:
                    st.markdown(f"### {home_score:.2f}")
                    st.markdown(f"### {away_score:.2f}")

                with col3:
                    if away_score > home_score:
                        st.markdown(f"### {away_emoji} **{away_label}**")
                    else:
                        st.markdown(f"### {away_emoji} {away_label}")

                # Show margin if not a tie
                if winner:
                    st.caption(f"*{label_owner(winner)} wins by {margin:.2f} points*")
                else:
                    st.caption("*TIE GAME*")

                st.markdown("---")

        # Week leaderboard
        st.subheader("Week Leaderboard")

        # Combine all scores
        all_scores = []
        for idx, game in week_matchups.iterrows():
            all_scores.append({
                'owner': game['owner_id_home'],
                'score': game['score_home'],
                'opponent': game['owner_id_away'],
                'opp_score': game['score_away'],
                'result': 'W' if game['score_home'] > game['score_away'] else ('L' if game['score_home'] < game['score_away'] else 'T')
            })
            all_scores.append({
                'owner': game['owner_id_away'],
                'score': game['score_away'],
                'opponent': game['owner_id_home'],
                'opp_score': game['score_home'],
                'result': 'W' if game['score_away'] > game['score_home'] else ('L' if game['score_away'] < game['score_home'] else 'T')
            })

        leaderboard = pd.DataFrame(all_scores).sort_values('score', ascending=False).reset_index(drop=True)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        leaderboard['margin'] = (leaderboard['score'] - leaderboard['opp_score']).round(2)

        display_leaderboard = leaderboard[['rank', 'owner', 'score', 'result', 'opponent', 'margin']].copy()
        display_leaderboard['owner'] = label_owner_series(display_leaderboard['owner'])
        display_leaderboard['opponent'] = label_owner_series(display_leaderboard['opponent'])
        display_leaderboard.columns = ['Rank', 'Owner', 'Score', 'Result', 'Opponent', 'Margin']
        display_leaderboard['Score'] = display_leaderboard['Score'].round(2)

        st.dataframe(
            display_leaderboard,
            hide_index=True,
            use_container_width=True,
            height=min(500, len(display_leaderboard) * 40 + 100)
        )

        # Export functionality
        st.markdown("###")
        csv = display_leaderboard.to_csv(index=False)
        st.download_button(
            label=f"📥 Export Week {selected_week} Results (CSV)",
            data=csv,
            file_name=f"cataneration_{selected_season}_week{selected_week}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    render_data_attribution(game_filter)
