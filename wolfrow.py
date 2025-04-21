import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.linear_model import LinearRegression
import random

# Constants and setup
CSV_PATH = "workouts.csv"
WEEK_TYPES = ["base", "threshold", "taper"]

# Initialize session state
def init_session():
    if "history" not in st.session_state:
        if os.path.exists(CSV_PATH):
            st.session_state.history = pd.read_csv(CSV_PATH, parse_dates=["date"])
        else:
            st.session_state.history = pd.DataFrame(columns=["date", "distance", "time", "pace", "week_type"])

    if "week_type" not in st.session_state:
        st.session_state.week_type = random.choice(WEEK_TYPES)

    if "model" not in st.session_state:
        st.session_state.model = LinearRegression()

init_session()

# App UI
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

# Pace prediction and fatigue detection
if len(st.session_state.history) >= 2:
    df = st.session_state.history.sort_values("date")
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["pace"].values
    st.session_state.model.fit(X, y)
    trend = st.session_state.model.predict(X)
    fatigue = y[-1] > trend[-1] * 1.05  # 5% slower than trend implies fatigue
    
    st.write("### Fatigue Prediction")
    st.metric("Latest pace", f"{y[-1]:.2f} sec/m")
    st.metric("Predicted trend pace", f"{trend[-1]:.2f} sec/m")
    st.warning("🚨 Signs of fatigue detected.") if fatigue else st.success("✅ No significant fatigue.")

# Reinforcement logic (simple reward: faster pace = reward)
if len(st.session_state.history) >= 3:
    last_two = st.session_state.history.tail(2)
    reward = (last_two.iloc[0]["pace"] - last_two.iloc[1]["pace"]) * 100  # reward = pace improvement
    if reward > 0.5:
        st.session_state.week_type = "threshold"
    elif reward < -0.5:
        st.session_state.week_type = "taper"
    else:
        st.session_state.week_type = "base"

# Workout suggestion
st.write("### Suggested Workout")
base = st.session_state.history["pace"].mean() if not st.session_state.history.empty else 2.0

if st.session_state.week_type == "threshold":
    suggestion = f"3 x 8 min @ {base + 0.2:.2f} sec/m"
elif st.session_state.week_type == "taper":
    suggestion = f"2 x 10 min @ {base + 0.4:.2f} sec/m, low rate"
else:
    suggestion = f"45 min steady @ {base + 0.3:.2f} sec/m"

st.info(suggestion)

# Graphs
st.write("### Pace Over Time")
if not st.session_state.history.empty:
    df = st.session_state.history.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["pace"], marker="o", label="Pace")
    ax.plot(df["date"], st.session_state.model.predict(np.arange(len(df)).reshape(-1, 1)), label="Trend")
    ax.set_ylabel("Pace (sec/m)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

st.write("---")
st.caption("All workouts saved to workouts.csv in GitHub-connected directory.")
