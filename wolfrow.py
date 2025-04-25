import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import requests
from sklearn.linear_model import LinearRegression
import random
import calendar

# Constants
CSV_PATH = "workouts.csv"
WEEK_TYPES = ["base", "threshold", "taper"]

# Initialize session state
def init_session():
    if "history" not in st.session_state:
        if os.path.exists(CSV_PATH):
            st.session_state.history = pd.read_csv(CSV_PATH)
            st.session_state.history["date"] = pd.to_datetime(st.session_state.history["date"], errors="coerce")
        else:
            st.session_state.history = pd.DataFrame(columns=["date", "distance", "time", "pace", "week_type"])
    if "week_type" not in st.session_state:
        st.session_state.week_type = random.choice(WEEK_TYPES)
    if "model" not in st.session_state:
        st.session_state.model = LinearRegression()

init_session()

# UI
st.title("🏋️ Adaptive Rowing Trainer")
st.subheader(f"This week is a **{st.session_state.week_type.upper()}** week")

# Workout logging
with st.form("log_workout"):
    st.write("### Log a new workout")
    date = st.date_input("Date", datetime.date.today())
    distance = st.number_input("Distance (meters)", min_value=100, step=100)
    time = st.number_input("Time (minutes)", min_value=1.0, step=0.5)
    submitted = st.form_submit_button("Log Workout")

    if submitted:
        pace = time * 60 / distance  # seconds per meter
        new_row = pd.DataFrame({
            "date": [date],
            "distance": [distance],
            "time": [time],
            "pace": [pace],
            "week_type": [st.session_state.week_type]
        })
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        st.session_state.history.to_csv(CSV_PATH, index=False)
        st.success("Workout logged!")

        # ✅ POST to FastAPI
        api_payload = {
            "date": date.isoformat(),
            "distance": distance,
            "time": time,
            "pace": pace,
            "week_type": st.session_state.week_type
        }
        try:
            response = requests.post("http://127.0.0.1:8000/log_workout", json=api_payload)
            st.success(f"📡 API response: {response.status_code} - {response.json()}")
        except Exception as e:
            st.error(f"API Error: {e}")

# Fatigue and prediction
if len(st.session_state.history) >= 2:
    df = st.session_state.history.copy().sort_values("date")
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["pace"].values
    st.session_state.model.fit(X, y)
    trend = st.session_state.model.predict(X)
    fatigue = y[-1] > trend[-1] * 1.05

    st.write("### Fatigue Prediction")
    st.metric("Latest pace", f"{y[-1]:.2f} sec/m")
    st.metric("Predicted trend pace", f"{trend[-1]:.2f} sec/m")
    if fatigue:
        st.warning("🚨 Signs of fatigue detected.")
    else:
        st.success("✅ No significant fatigue.")

# History
st.write("### 📈 Workout History")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history.sort_values("date", ascending=False).reset_index(drop=True))
else:
    st.info("No workouts logged yet.")

# Week type adaptation
if len(st.session_state.history) >= 3:
    last_two = st.session_state.history.tail(2)
    reward = (last_two.iloc[0]["pace"] - last_two.iloc[1]["pace"]) * 100
    if reward > 0.5:
        st.session_state.week_type = "threshold"
    elif reward < -0.5:
        st.session_state.week_type = "taper"
    else:
        st.session_state.week_type = "base"

# Workout Suggestion
st.write("### Suggested Workout")
base = st.session_state.history["pace"].mean() if not st.session_state.history.empty else 2.0
suggestion = {
    "threshold": f"3 x 8 min @ {base + 0.2:.2f} sec/m",
    "taper": f"2 x 10 min @ {base + 0.4:.2f} sec/m, low rate",
    "base": f"45 min steady @ {base + 0.3:.2f} sec/m"
}[st.session_state.week_type]
st.info(suggestion)

# Graph
st.write("### Pace Over Time")
if not st.session_state.history.empty:
    df = st.session_state.history.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["pace"], marker="o", label="Pace")
    if len(df) >= 2:
        trend_line = st.session_state.model.predict(np.arange(len(df)).reshape(-1, 1))
        ax.plot(df["date"], trend_line, label="Trend")
    ax.set_ylabel("Pace (sec/m)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

# Heatmap
st.write("### Weekly Heatmap of Training Time")
df = st.session_state.history.copy()
if not df.empty:
    df["weekday"] = df["date"].dt.day_name()
    pivot = df.groupby("weekday")["time"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Total Time (minutes)")
    st.pyplot(fig)

# Weekly report
st.write("### 📊 Weekly Report")
this_week = df[df["date"] >= (datetime.datetime.now() - datetime.timedelta(days=7))]
if not this_week.empty:
    total_time = this_week["time"].sum()
    avg_pace = this_week["pace"].mean()
    st.markdown(f"**Total Time**: {total_time:.1f} minutes")
    st.markdown(f"**Average Pace**: {avg_pace:.2f} sec/m")
else:
    st.info("No workouts this week.")

# VO2max estimate
st.write("### 🧠 VO2max Estimate")
if not df.empty:
    row = df.iloc[-1]
    speed = row["distance"] / (row["time"] * 60)
    vo2max = 15.2 * speed ** 0.65
    st.metric("Estimated VO2max", f"{vo2max:.2f} ml/kg/min")

# Export buttons (placeholder)
st.write("### 📤 Export Options")
col1, col2 = st.columns(2)
with col1:
    if st.button("Export to Google Calendar"):
        st.info("🗓️ Export to Google Calendar coming soon!")
with col2:
    if st.button("Export to Strava"):
        st.info("🚴 Export to Strava coming soon!")

st.write("---")
st.caption("All workouts saved to workouts.csv and synced to API.")

