"""
Live Win Probability Replay
-----------------------------
Replay any 2015/2016 match (Premier League, La Liga, Serie A) minute-by-minute,
showing how a live win-probability model's predictions evolved as the match
unfolded -- starting from a pre-match Elo-based prior and updating on score,
possession, xG, red cards, and shot momentum.

Expects these files in a `data/` folder next to this script:
  - snapshot_features_full.parquet   (from notebook 03's final save)
  - xgb_live_model.json              (fitted XGBoost model, native JSON format --
                                       version-stable, unlike pickle)
  - live_model_feature_cols.json     (exact feature column order)
  - match_events.parquet             (goals, own goals, red cards: who and when)

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from xgboost import XGBClassifier

st.set_page_config(page_title="Live Win Probability Replay", page_icon="⚽", layout="wide")


@st.cache_data
def load_snapshot_data():
    return pd.read_parquet("data/snapshot_features_full.parquet")


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("data/xgb_live_model.json")
    with open("data/live_model_feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


@st.cache_data
def load_match_events():
    """Goals, own goals and red cards for every match. Returns None if the file is missing,
    so the app still works (without event markers) rather than crashing."""
    try:
        return pd.read_parquet("data/match_events.parquet")
    except FileNotFoundError:
        return None


HOME_EVENT_COLOR = "#E0A800"   # darker gold than the probability line, so it reads on white
AWAY_EVENT_COLOR = "#132257"
RED_CARD_COLOR = "#D62728"
LABEL_BASE_Y = 1.06     # height of the first label row (just above the 100% line)
LABEL_ROW_STEP = 0.065  # vertical gap between label rows
MAX_LABEL_ROWS = 5
ASSUMED_PLOT_PX = 700   # conservative plot width, so labels also fit on tablets and small windows


def event_label(ev):
    """Short on-chart label, e.g. a goal icon + scorer + minute, or a red-card icon + player + minute."""
    if ev.event_type == "Red card":
        return f"🟥 {ev.player} {ev.minute}'"
    suffix = " (OG)" if ev.event_type == "Own goal" else ""
    return f"⚽ {ev.player}{suffix} {ev.minute}'"


def _label_extent(ev, max_minute, px_per_min):
    """Approximate horizontal span (in match minutes) a label occupies on the chart."""
    width_px = 16 + 6.3 * (len(event_label(ev).split(" ", 1)[1]) + 1) + 8   # emoji + text + padding
    w = width_px / px_per_min
    if ev.minute < 12:
        return ev.minute, ev.minute + w
    if ev.minute > max_minute - 12:
        return ev.minute - w, ev.minute
    return ev.minute - w / 2, ev.minute + w / 2


def assign_label_rows(events, max_minute):
    """Greedy lane assignment: each label goes in the first row where it doesn't collide with the
    previous label in that row. Computed from the match's FULL event list (not just the events
    visible so far), so a label never jumps rows while the slider moves."""
    px_per_min = ASSUMED_PLOT_PX / (max_minute + 2)
    row_ends = [-1e9] * MAX_LABEL_ROWS
    rows = []
    for ev in events.itertuples():
        start, end = _label_extent(ev, max_minute, px_per_min)
        row = next((k for k in range(MAX_LABEL_ROWS) if start > row_ends[k] + 0.5), None)
        if row is None:  # every row busy: use whichever frees up first
            row = min(range(MAX_LABEL_ROWS), key=lambda k: row_ends[k])
        row_ends[row] = max(row_ends[row], end)
        rows.append(row)
    return rows


def make_figure(timeline, events, minute, home_team, away_team, max_minute):
    visible = timeline[timeline["snapshot_minute"] <= minute]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=visible["snapshot_minute"], y=visible["prob_home_win_live"],
                              name=f"{home_team} (Home)", line=dict(color="#FFCC00", width=3)))
    fig.add_trace(go.Scatter(x=visible["snapshot_minute"], y=visible["prob_draw_live"],
                              name="Draw", line=dict(color="gray", width=3)))
    fig.add_trace(go.Scatter(x=visible["snapshot_minute"], y=visible["prob_away_win_live"],
                              name=f"{away_team} (Away)", line=dict(color="#132257", width=3)))

    # Mark the pre-match prior at minute 0 for reference
    fig.add_trace(go.Scatter(
        x=[0], y=[timeline["prob_home_win"].iloc[0]],
        mode="markers", marker=dict(symbol="x", size=12, color="#FFCC00"),
        name="Pre-match prior (Home)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[timeline["prob_away_win"].iloc[0]],
        mode="markers", marker=dict(symbol="x", size=12, color="#132257"),
        name="Pre-match prior (Away)", showlegend=True,
    ))

    # Goals and red cards: dotted line through the plot + a label in the band above it.
    # Event minutes match the chart's x-axis (the first minute whose probabilities include the
    # event), so each marker sits exactly where the curve reacts. Events appear as the slider
    # reaches them, like a live replay. Label rows are assigned from the full match (see
    # assign_label_rows) so a label never jumps rows as the slider moves.
    label_rows = assign_label_rows(events, max_minute) if events is not None and len(events) else []
    n_rows = (max(label_rows) + 1) if label_rows else 0

    if events is not None:
        for i, ev in enumerate(events.itertuples()):
            if ev.minute > minute:
                continue
            if ev.event_type == "Red card":
                color = RED_CARD_COLOR
            else:
                color = HOME_EVENT_COLOR if ev.team == home_team else AWAY_EVENT_COLOR
            fig.add_shape(type="line", x0=ev.minute, x1=ev.minute, y0=0, y1=1.0,
                          line=dict(color=color, width=1.5, dash="dot"))
            anchor = "left" if ev.minute < 12 else "right" if ev.minute > max_minute - 12 else "center"
            fig.add_annotation(x=ev.minute, y=LABEL_BASE_Y + LABEL_ROW_STEP * label_rows[i], text=event_label(ev),
                               showarrow=False, xanchor=anchor, font=dict(size=11, color="#222"),
                               bgcolor="rgba(255,255,255,0.85)")

    fig.update_layout(
        xaxis_title="Match Minute",
        yaxis_title="Win Probability",
        yaxis=dict(range=[0, LABEL_BASE_Y + LABEL_ROW_STEP * max(n_rows - 1, 0) + 0.045],
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], tickformat=".0%"),
        xaxis=dict(range=[-1, max_minute + 1]),
        height=470 + 24 * max(n_rows, 1),
        hovermode="x unified",
    )
    return fig


def events_table(events, minute):
    """Plain-language list of the events that have happened so far."""
    shown = events[events["minute"] <= minute]
    rows = []
    for ev in shown.itertuples():
        if ev.event_type == "Red card":
            what = "🟥 Red card (second yellow)" if ev.detail == "Second Yellow" else "🟥 Red card"
            who = ev.player
        elif ev.event_type == "Own goal":
            what, who = "⚽ Own goal", f"{ev.player} (OG)"
        else:
            what = "⚽ Goal (penalty)" if ev.detail == "Penalty" else "⚽ Goal"
            who = ev.player
        rows.append({"Minute": f"{ev.minute}'", "Event": what, "Player": who, "Team": ev.team})
    return pd.DataFrame(rows)


# The Watford 1-2 Tottenham match (Dec 28, 2015) used as the project's showcase example.
# Defaulting by exact match_id rather than team names, since these two clubs met twice
# in the dataset and a name-based lookup can silently pick the wrong fixture.
SHOWCASE_MATCH_ID = 3754291


def get_match_list(snapshot_df):
    # Final score per match = the last snapshot's cumulative goal counts
    final_scores = (
        snapshot_df.sort_values("snapshot_minute")
        .groupby("match_id")
        .agg(home_team=("home_team", "first"), away_team=("away_team", "first"),
             home_goals=("home_goals_cum", "last"), away_goals=("away_goals_cum", "last"))
        .reset_index()
    )
    final_scores["label"] = (
        final_scores["home_team"] + " " + final_scores["home_goals"].astype(int).astype(str)
        + " - " + final_scores["away_goals"].astype(int).astype(str) + " " + final_scores["away_team"]
    )
    return final_scores.sort_values("label").reset_index(drop=True)


def build_probability_timeline(snapshot_df, match_id, model, feature_cols):
    match_data = snapshot_df[snapshot_df["match_id"] == match_id].sort_values("snapshot_minute").copy()
    X = match_data[feature_cols]
    probs = model.predict_proba(X)
    # Encoded class order confirmed in this project: 0=away_win, 1=draw, 2=home_win
    match_data["prob_away_win_live"] = probs[:, 0]
    match_data["prob_draw_live"] = probs[:, 1]
    match_data["prob_home_win_live"] = probs[:, 2]
    return match_data


def main():
    st.title("⚽ Live Win Probability Replay")
    st.markdown(
        "Replay a real 2015/2016 match and watch win probability update live -- "
        "starting from a pre-match Elo-based prior, then reacting to goals, red "
        "cards, possession swings, and shot quality as the match unfolds."
    )

    try:
        snapshot_df = load_snapshot_data()
        model, feature_cols = load_model()
        match_events = load_match_events()
    except FileNotFoundError as e:
        st.error(
            f"Couldn't find required data file: {e}. "
            "Make sure `data/snapshot_features_full.parquet`, `data/xgb_live_model.json`, "
            "and `data/live_model_feature_cols.json` are present next to this app."
        )
        st.stop()

    match_list = get_match_list(snapshot_df)
    labels = match_list["label"].tolist()

    default_index = 0
    showcase_rows = match_list[match_list["match_id"] == SHOWCASE_MATCH_ID]
    if len(showcase_rows) > 0:
        default_index = labels.index(showcase_rows["label"].iloc[0])

    selected_label = st.selectbox("Choose a match", labels, index=default_index)
    selected_match_id = match_list[match_list["label"] == selected_label]["match_id"].iloc[0]

    timeline = build_probability_timeline(snapshot_df, selected_match_id, model, feature_cols)
    home_team = timeline["home_team"].iloc[0]
    away_team = timeline["away_team"].iloc[0]
    max_minute = int(timeline["snapshot_minute"].max())

    minute = st.slider("Match minute", min_value=0, max_value=max_minute, value=max_minute)

    current = timeline[timeline["snapshot_minute"] <= minute].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{home_team} (Home)", f"{current['prob_home_win_live']*100:.1f}%")
    col2.metric("Draw", f"{current['prob_draw_live']*100:.1f}%")
    col3.metric(f"{away_team} (Away)", f"{current['prob_away_win_live']*100:.1f}%")

    st.caption(
        f"Score at minute {minute}: {home_team} {int(current['home_goals_cum'])} - "
        f"{int(current['away_goals_cum'])} {away_team}"
        + (f" | Numerical advantage: {home_team if current['man_advantage'] > 0 else away_team}"
           if current["man_advantage"] != 0 else "")
    )

    if match_events is None:
        st.info("Goal and red-card markers are unavailable (data/match_events.parquet not found).")
        events = None
    else:
        events = match_events[match_events["match_id"] == selected_match_id].reset_index(drop=True)

    st.plotly_chart(make_figure(timeline, events, minute, home_team, away_team, max_minute),
                    use_container_width=True)

    if events is not None:
        st.subheader("Match events")
        table = events_table(events, minute)
        if table.empty:
            st.caption("No goals or red cards yet at this point in the match.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Model: XGBoost classifier trained on live match-state snapshots (score, "
        "possession, cumulative xG, red cards, shot momentum), using an Elo-based "
        "pre-match prior as a starting point. The prior's influence was measured to "
        "improve early-match predictions roughly 5x more than late-match predictions, "
        "confirming it anchors the model when live evidence is thin and cedes "
        "influence as the match progresses. Data: StatsBomb open data, 2015/2016 "
        "season (Premier League, La Liga, Serie A)."
    )


if __name__ == "__main__":
    main()
