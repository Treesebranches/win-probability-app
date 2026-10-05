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

    fig.update_layout(
        xaxis_title="Match Minute",
        yaxis_title="Win Probability",
        yaxis_range=[0, 1],
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

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
