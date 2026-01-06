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
def get_league_benchmarks() -> dict:
    """Calculate league-wide benchmarks for context."""
    df = load_all_time_stats()
    matchups_df = load_matchups()

    return {
        'avg_win_pct': df['win_pct'].mean(),
        'avg_ppg': df['avg_points_for'].mean(),
        'total_games': len(matchups_df) // 2,
        'seasons': matchups_df['season'].nunique(),
        'active_owners': len(df)
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

def render_data_attribution():
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

    st.markdown("---")
    st.markdown(
        f"""
        <div class="footer">
        <strong>Data Sources:</strong> MyFantasyLeague (2013-2019), Fleaflicker (2020-2024), Sleeper (2025)<br>
        <strong>Last Updated:</strong> {last_updated or 'Unknown'}<br>
        <strong>Total Games:</strong> 1,160 | <strong>Seasons:</strong> 13 (2013-2025) | <strong>Methodology:</strong> All regular season and playoff matchups included
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

page = st.sidebar.radio(
    "Navigate",
    [
        "All-Time Leaderboard",
        "Owner Profile",
        "Head-to-Head Explorer",
        "Weekly Timeline"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Quick stats in sidebar
benchmarks = get_league_benchmarks()
st.sidebar.markdown("### League Overview")
st.sidebar.metric("Total Games", f"{benchmarks['total_games']:,}")
st.sidebar.metric("Active Owners", benchmarks['active_owners'])
st.sidebar.metric("Seasons", benchmarks['seasons'])
st.sidebar.metric("Avg Win %", f"{benchmarks['avg_win_pct']*100:.1f}%")

# ==============================================================================
# PAGE 1: ALL-TIME LEADERBOARD
# ==============================================================================

if page == "All-Time Leaderboard":
    st.title("All-Time Leaderboard")
    st.caption("Comprehensive performance rankings across all platforms and seasons")

    # Load data
    df = load_all_time_stats()
    benchmarks = get_league_benchmarks()

    # Key metrics at top (Bloomberg style: 4 key numbers)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_owner = df.iloc[0]
        st.metric(
            "League Leader",
            top_owner['owner_id'],
            format_pct(top_owner['win_pct'])
        )
    with col2:
        best_scorer = df.loc[df['avg_points_for'].idxmax()]
        st.metric(
            "Highest Avg PPG",
            best_scorer['owner_id'],
            f"{best_scorer['avg_points_for']:.1f}"
        )
    with col3:
        most_games = df.loc[df['games'].idxmax()]
        st.metric(
            "Most Games",
            most_games['owner_id'],
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
    tab1, tab2, tab3 = st.tabs(["📊 Rankings Table", "📈 Win % Chart", "🎯 Points Scoring Chart"])

    with tab1:
        # Prepare display dataframe
        display_df = df.copy()
        display_df['rank'] = range(1, len(display_df) + 1)
        display_df['record'] = display_df.apply(
            lambda x: format_record(x['wins'], x['losses'], x['ties']), axis=1
        )

        # Style the dataframe
        styled_df = display_df[[
            'rank', 'owner_id', 'games', 'record', 'win_pct',
            'points_for', 'points_against', 'avg_points_for', 'avg_points_against'
        ]].copy()

        styled_df.columns = [
            'Rank', 'Owner', 'GP', 'Record', 'Win %',
            'PF', 'PA', 'PPG', 'PAPG'
        ]

        # Format numbers
        styled_df['Win %'] = styled_df['Win %'].apply(lambda x: f"{x*100:.1f}%")
        styled_df['PF'] = styled_df['PF'].apply(lambda x: f"{x:,.1f}")
        styled_df['PA'] = styled_df['PA'].apply(lambda x: f"{x:,.1f}")
        styled_df['PPG'] = styled_df['PPG'].apply(lambda x: f"{x:.2f}")
        styled_df['PAPG'] = styled_df['PAPG'].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=min(800, len(styled_df) * 40 + 100)
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
                st.markdown(f"**{owner['owner_id']}**")
                st.caption(f"{format_record(owner['wins'], owner['losses'], owner['ties'])} ({format_pct(owner['win_pct'])})")
                st.caption(f"{owner['avg_points_for']:.2f} PPG • {int(owner['games'])} games")

    with tab2:
        # Win percentage bar chart with league average benchmark
        chart_df = df.head(15).copy()  # Top 15 for readability
        fig = create_bar_chart(
            chart_df,
            x='win_pct',
            y='owner_id',
            title='Win Percentage Leaders (Min. Games Played)',
            benchmark=benchmarks['avg_win_pct']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("*Chart shows top 15 owners by win percentage. Dashed line indicates league average.*")

    with tab3:
        # Points per game comparison
        chart_df = df.head(15).copy()
        fig = create_bar_chart(
            chart_df,
            x='avg_points_for',
            y='owner_id',
            title='Average Points Per Game Leaders',
            benchmark=benchmarks['avg_ppg']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("*Chart shows top 15 owners by average points per game. Dashed line indicates league average.*")

    # Export functionality
    st.markdown("###")
    csv = styled_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Leaderboard (CSV)",
        data=csv,
        file_name=f"cataneration_all_time_leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    render_data_attribution()

# ==============================================================================
# PAGE 2: OWNER PROFILE
# ==============================================================================

elif page == "Owner Profile":
    st.title("Owner Profile")
    st.caption("Deep dive into individual season-by-season performance")

    # Load data
    season_df = load_season_stats()
    all_time_df = load_all_time_stats()
    benchmarks = get_league_benchmarks()

    # Owner selector with search
    owners = sorted(all_time_df['owner_id'].unique())
    selected_owner = st.selectbox(
        "Select Owner",
        owners,
        help="Choose an owner to view their complete performance history"
    )

    if selected_owner:
        # Get all-time stats for this owner
        owner_stats = all_time_df[all_time_df['owner_id'] == selected_owner].iloc[0]
        owner_rank = all_time_df[all_time_df['owner_id'] == selected_owner].index[0] + 1

        # Header with owner name and rank
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {selected_owner}")
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
                display_seasons['win_pct_display'] = display_seasons['win_pct'].apply(format_pct)

                # Calculate vs league average for each season
                league_avg_by_season = season_df.groupby('season')['win_pct'].mean()
                display_seasons['vs_avg'] = display_seasons.apply(
                    lambda x: f"{(x['win_pct'] - league_avg_by_season.get(x['season'], 0.5))*100:+.1f}%",
                    axis=1
                )

                styled_seasons = display_seasons[[
                    'season', 'record', 'win_pct_display', 'vs_avg',
                    'points_for', 'points_against'
                ]].copy()

                styled_seasons.columns = [
                    'Season', 'Record', 'Win %', 'vs League Avg', 'PF', 'PA'
                ]

                styled_seasons['PF'] = styled_seasons['PF'].apply(lambda x: f"{x:.1f}")
                styled_seasons['PA'] = styled_seasons['PA'].apply(lambda x: f"{x:.1f}")

                st.dataframe(
                    styled_seasons,
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, len(styled_seasons) * 40 + 100)
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
                st.info(f"No season data found for {selected_owner}")

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
                    name=selected_owner,
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{selected_owner}</b><br>Season: %{{x}}<br>Win %: %{{y:.1f}}%<extra></extra>'
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
                    st.metric("Luck Factor",
                             format_pct(abs(luck_factor)),
                             "Lucky" if luck_factor > 0 else "Unlucky")
                    st.caption("*Based on points scored vs allowed. Positive = won more than expected.*")

        # Export functionality
        st.markdown("###")
        if len(owner_seasons) > 0:
            csv = styled_seasons.to_csv(index=False)
            st.download_button(
                label=f"📥 Export {selected_owner} History (CSV)",
                data=csv,
                file_name=f"cataneration_{selected_owner}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    render_data_attribution()

# ==============================================================================
# PAGE 3: HEAD-TO-HEAD EXPLORER
# ==============================================================================

elif page == "Head-to-Head Explorer":
    st.title("Head-to-Head Explorer")
    st.caption("Complete matchup history between any two owners")

    # Load data
    h2h_df = load_h2h_stats()
    all_time_df = load_all_time_stats()
    matchups_df = load_matchups()

    # Owner selectors
    owners = sorted(all_time_df['owner_id'].unique())

    col1, col2 = st.columns(2)
    with col1:
        owner_a = st.selectbox("Owner A", owners, key="owner_a")
    with col2:
        available_opponents = [o for o in owners if o != owner_a]
        owner_b = st.selectbox("Owner B", available_opponents, key="owner_b")

    if owner_a and owner_b:
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
            st.markdown(f"## {owner_a} vs {owner_b}")
            st.caption(f"All-time series: {total_games} games")
            st.markdown("###")

            # Win-Loss visualization
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"### {owner_a}")
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
                st.markdown(f"### {owner_b}")
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

                    display_games = games.copy()

                    # Determine winner and loser for each game
                    def format_matchup(row):
                        if row['owner_id_home'] == owner_a:
                            home = owner_a
                            away = owner_b
                            home_score = row['score_home']
                            away_score = row['score_away']
                        else:
                            home = owner_b
                            away = owner_a
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
                    display_games['playoffs'] = display_games['is_playoffs'].apply(lambda x: '🏆' if x else '')
                    display_games['winner'] = display_games.apply(
                        lambda x: x['owner_id_home'] if x['score_home'] > x['score_away']
                                 else (x['owner_id_away'] if x['score_away'] > x['score_home'] else 'TIE'),
                        axis=1
                    )

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
                            st.caption(f"**{season}:** {owner_a} {a_season_wins}-{b_season_wins} {owner_b}")

                    with col2:
                        st.markdown("##### Context")

                        playoff_games = len(games[games['is_playoffs'] == True])
                        regular_games = len(games[games['is_playoffs'] == False])

                        st.metric("Playoff Matchups", playoff_games)
                        st.metric("Regular Season", regular_games)

                        # Blowouts (>20 point margin)
                        games['margin'] = abs(games['score_home'] - games['score_away'])
                        blowouts = len(games[games['margin'] >= 20])
                        close_games = len(games[games['margin'] <= 5])

                        st.metric("Blowouts (20+ pts)", blowouts)
                        st.metric("Close Games (≤5 pts)", close_games)

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
                        name=owner_a,
                        line=dict(color=COLORS['primary'], width=2.5),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{owner_a}</b><br>Game %{{x}}<br>Score: %{{y:.2f}}<extra></extra>'
                    ))

                    fig.add_trace(go.Scatter(
                        x=game_nums,
                        y=owner_b_scores,
                        mode='lines+markers',
                        name=owner_b,
                        line=dict(color=COLORS['secondary'], width=2.5),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{owner_b}</b><br>Game %{{x}}<br>Score: %{{y:.2f}}<extra></extra>'
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
            st.info(f"No head-to-head record found between {owner_a} and {owner_b}")

    render_data_attribution()

# ==============================================================================
# PAGE 4: WEEKLY TIMELINE
# ==============================================================================

elif page == "Weekly Timeline":
    st.title("Weekly Timeline")
    st.caption("Explore every week of every season")

    # Load data
    matchups_df = load_matchups()

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
        is_playoffs = week_matchups.iloc[0]['is_playoffs']

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

                with col1:
                    if home_score > away_score:
                        st.markdown(f"### {home_emoji} **{game['owner_id_home']}**")
                    else:
                        st.markdown(f"### {home_emoji} {game['owner_id_home']}")

                with col2:
                    st.markdown(f"### {home_score:.2f}")
                    st.markdown(f"### {away_score:.2f}")

                with col3:
                    if away_score > home_score:
                        st.markdown(f"### {away_emoji} **{game['owner_id_away']}**")
                    else:
                        st.markdown(f"### {away_emoji} {game['owner_id_away']}")

                # Show margin if not a tie
                if winner:
                    st.caption(f"*{winner} wins by {margin:.2f} points*")
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

    render_data_attribution()
