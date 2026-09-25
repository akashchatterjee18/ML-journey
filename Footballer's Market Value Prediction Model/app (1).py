import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("football_market_value_model.pkl")
scaler = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Football Player Market Value Prediction by Akash Chatterjee",
    page_icon="⚽",
    layout="wide"
)


st.title("Football Player Market Value Prediction by Akash Chatterjee")

st.markdown(
    "Enter the player's attributes below to estimate their "
    "football market value."
)

st.header("Player Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=50,
        value=22
    )

    overall = st.number_input(
        "Overall Rating",
        min_value=0,
        max_value=100,
        value=70
    )

    potential = st.number_input(
        "Potential",
        min_value=0,
        max_value=100,
        value=75
    )

with col2:
    special = st.number_input(
        "Special",
        min_value=0,
        max_value=1000,
        value=500
    )

    acceleration = st.number_input(
        "Acceleration",
        min_value=0,
        max_value=100,
        value=70
    )

    sprint_speed = st.number_input(
        "Sprint Speed",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    agility = st.number_input(
        "Agility",
        min_value=0,
        max_value=100,
        value=70
    )

    balance = st.number_input(
        "Balance",
        min_value=0,
        max_value=100,
        value=70
    )

    stamina = st.number_input(
        "Stamina",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Physical attributes
# --------------------------------------------------

st.header("Physical Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    jumping = st.number_input(
        "Jumping",
        min_value=0,
        max_value=100,
        value=70
    )

    strength = st.number_input(
        "Strength",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    ball_control = st.number_input(
        "Ball Control",
        min_value=0,
        max_value=100,
        value=70
    )

    dribbling = st.number_input(
        "Dribbling",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    composure = st.number_input(
        "Composure",
        min_value=0,
        max_value=100,
        value=70
    )

    reactions = st.number_input(
        "Reactions",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Passing attributes
# --------------------------------------------------

st.header("Passing Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    vision = st.number_input(
        "Vision",
        min_value=0,
        max_value=100,
        value=70
    )

    short_passing = st.number_input(
        "Short Passing",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    long_passing = st.number_input(
        "Long Passing",
        min_value=0,
        max_value=100,
        value=70
    )

    crossing = st.number_input(
        "Crossing",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    curve = st.number_input(
        "Curve",
        min_value=0,
        max_value=100,
        value=70
    )

    free_kick_accuracy = st.number_input(
        "Free Kick Accuracy",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Attacking attributes
# --------------------------------------------------

st.header("Attacking Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    finishing = st.number_input(
        "Finishing",
        min_value=0,
        max_value=100,
        value=70
    )

    shot_power = st.number_input(
        "Shot Power",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    long_shots = st.number_input(
        "Long Shots",
        min_value=0,
        max_value=100,
        value=70
    )

    positioning = st.number_input(
        "Positioning",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    volleys = st.number_input(
        "Volleys",
        min_value=0,
        max_value=100,
        value=70
    )

    heading_accuracy = st.number_input(
        "Heading Accuracy",
        min_value=0,
        max_value=100,
        value=70
    )

    penalties = st.number_input(
        "Penalties",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Defending attributes
# --------------------------------------------------

st.header("Defending Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    aggression = st.number_input(
        "Aggression",
        min_value=0,
        max_value=100,
        value=70
    )

    interceptions = st.number_input(
        "Interceptions",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    marking = st.number_input(
        "Marking",
        min_value=0,
        max_value=100,
        value=70
    )

    standing_tackle = st.number_input(
        "Standing Tackle",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    sliding_tackle = st.number_input(
        "Sliding Tackle",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Goalkeeping attributes
# --------------------------------------------------

st.header("Goalkeeping Attributes")

col1, col2, col3 = st.columns(3)

with col1:
    gk_diving = st.number_input(
        "GK Diving",
        min_value=0,
        max_value=100,
        value=70
    )

    gk_handling = st.number_input(
        "GK Handling",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    gk_kicking = st.number_input(
        "GK Kicking",
        min_value=0,
        max_value=100,
        value=70
    )

    gk_positioning = st.number_input(
        "GK Positioning",
        min_value=0,
        max_value=100,
        value=70
    )

with col3:
    gk_reflexes = st.number_input(
        "GK Reflexes",
        min_value=0,
        max_value=100,
        value=70
    )


# --------------------------------------------------
# Preferred position
# --------------------------------------------------

st.header("Preferred Position")

position = st.selectbox(
    "Select Preferred Position",
    [
        "GK",
        "CB",
        "LB",
        "RB",
        "LWB",
        "RWB",
        "CDM",
        "CM",
        "CAM",
        "LM",
        "RM",
        "LW",
        "RW",
        "LF",
        "RF",
        "CF",
        "ST",
        "LS",
        "RS",
        "LCB",
        "RCB",
        "LCM",
        "RCM",
        "LDM",
        "RDM",
        "LAM",
        "RAM"
    ]
)


# --------------------------------------------------
# Elite feature
# --------------------------------------------------

elite = int(overall >= 90)


# --------------------------------------------------
# Create input dictionary
# --------------------------------------------------

input_data = {
    "Age": age,
    "Overall": overall,
    "Potential": potential,
    "Special": special,
    "Elite": elite,

    "Acceleration": acceleration,
    "Sprint speed": sprint_speed,
    "Agility": agility,
    "Balance": balance,
    "Jumping": jumping,
    "Stamina": stamina,
    "Strength": strength,

    "Ball control": ball_control,
    "Dribbling": dribbling,
    "Composure": composure,
    "Reactions": reactions,

    "Vision": vision,
    "Short passing": short_passing,
    "Long passing": long_passing,
    "Crossing": crossing,
    "Curve": curve,
    "Free kick accuracy": free_kick_accuracy,

    "Finishing": finishing,
    "Shot power": shot_power,
    "Long shots": long_shots,
    "Positioning": positioning,
    "Volleys": volleys,
    "Heading accuracy": heading_accuracy,
    "Penalties": penalties,

    "Aggression": aggression,
    "Interceptions": interceptions,
    "Marking": marking,
    "Standing tackle": standing_tackle,
    "Sliding tackle": sliding_tackle,

    "GK diving": gk_diving,
    "GK handling": gk_handling,
    "GK kicking": gk_kicking,
    "GK positioning": gk_positioning,
    "GK reflexes": gk_reflexes,
}


# --------------------------------------------------
# Add preferred-position features
# --------------------------------------------------

positions = [
    "GK",
    "CB",
    "LB",
    "RB",
    "LWB",
    "RWB",
    "CDM",
    "CM",
    "CAM",
    "LM",
    "RM",
    "LW",
    "RW",
    "LF",
    "RF",
    "CF",
    "ST",
    "LS",
    "RS",
    "LCB",
    "RCB",
    "LCM",
    "RCM",
    "LDM",
    "RDM",
    "LAM",
    "RAM"
]

for pos in positions:
    input_data[f"{pos}_preferred_position"] = int(position == pos)


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

input_df = pd.DataFrame([input_data])


# --------------------------------------------------
# Match training columns
# --------------------------------------------------

for column in expected_columns:
    if column not in input_df.columns:
        input_df[column] = 0

input_df = input_df[expected_columns]


# --------------------------------------------------
# Scale only columns used during training
# --------------------------------------------------

scaled_columns = scaler.feature_names_in_

input_scaled = input_df.copy()

input_scaled[scaled_columns] = scaler.transform(
    input_df[scaled_columns]
)


# --------------------------------------------------
# Final feature order
# --------------------------------------------------

input_scaled = input_scaled[expected_columns]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Market Value"):

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"Predicted Market Value: €{prediction:,.2f}"
    )
