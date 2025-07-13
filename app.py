import streamlit as st
import pandas as pd
import joblib
import json

# Load saved files
model = joblib.load('vehicle_recommender_model.pkl')
scaler = joblib.load('vehicle_scaler.pkl')
with open('x_columns.json') as f:
    x_columns = json.load(f)
df_full = pd.read_csv('preprocessed_dataset.csv')

# App title
st.title("🚗 AI Vehicle Recommendation System")
st.write("Provide your preferred vehicle features below to get tailored recommendations.")

# ---------- User Inputs with Descriptions ----------
make = st.multiselect('Make (Brand of vehicle, e.g. Ford, BMW)', df_full['make'].dropna().unique())
year = st.number_input('Year (Manufacturing year)', min_value=1990, max_value=2030, value=2000)
model_name = st.text_input('Model (Specific model, optional)', '')
engine = st.text_input('Engine (Engine type, optional)', '')
cylinders = st.number_input('Cylinders (Number of cylinders in engine)', min_value=1, max_value=16, value=4)
fuel = st.selectbox('Fuel Type (Type of fuel used)', df_full['fuel'].dropna().unique())
mileage = st.number_input('Mileage (in miles, lower is better)', min_value=0.0, value=10.0)
transmission = st.text_input('Transmission (Type, e.g. Automatic, Manual, optional)', '')
trim = st.text_input('Trim (Variant or edition, optional)', '')
body = st.multiselect('Body Type (e.g. Sedan, SUV)', df_full['body'].dropna().unique())
doors = st.number_input('Doors (Number of doors)', min_value=1, max_value=8, value=4)
exterior_color = st.text_input('Exterior Color (optional)', '')
interior_color = st.text_input('Interior Color (optional)', '')
drivetrain = st.selectbox('Drivetrain (Drive configuration)', df_full['drivetrain'].dropna().unique())

# ---------- Recommend Button ----------
if st.button('🔍 Recommend Vehicles'):
    input_dict = {col: 0 for col in x_columns}

    # Numerical inputs
    for col in ['year', 'cylinders', 'mileage', 'doors']:
        if col in x_columns:
            input_dict[col] = locals()[col]

    # Multi-select makes
    for mk in make:
        make_col = f"make_{mk}"
        if make_col in x_columns:
            input_dict[make_col] = 1

    # Multi-select bodies
    for bd in body:
        body_col = f"body_{bd}"
        if body_col in x_columns:
            input_dict[body_col] = 1

    # Single select inputs
    for col_name, col_value in {
        'fuel': fuel,
        'drivetrain': drivetrain
    }.items():
        col_full = f"{col_name}_{col_value}"
        if col_full in x_columns:
            input_dict[col_full] = 1

    # Frequency encoded features with fallback
    freq_enc_cols = ['model_freq_enc', 'engine_freq_enc', 'trim_freq_enc',
                     'exterior_color_freq_enc', 'interior_color_freq_enc']
    for col in freq_enc_cols:
        base_col = col.replace('_freq_enc', '')
        input_dict[col] = 0
        if (col in x_columns) and (base_col in df_full.columns):
            user_val = locals().get(base_col, '')
            temp_df = df_full[df_full[base_col] == user_val]
            if not temp_df.empty and col in temp_df.columns:
                temp_mean = temp_df[col].mean()
                if pd.notnull(temp_mean):
                    input_dict[col] = temp_mean

    # Transmission simplified
    if 'transmission_simplified' in x_columns:
        input_dict['transmission_simplified'] = df_full['transmission_simplified'].mode()[0] if 'transmission_simplified' in df_full.columns else 0

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Recommendations
    distances, indices = model.kneighbors(input_scaled)

    # ---------- Display Results Professionally ----------
    st.markdown("## 🎯 **Top 5 Recommended Vehicles for You:**")
    for idx in indices[0]:
        vehicle = df_full.iloc[idx]

        st.markdown(f"""
        ### **{vehicle['year']} {vehicle['make']} {vehicle['model']}**
        - **Price:** ${vehicle['price']}
        - **Engine:** {vehicle['engine']}
        - **Cylinders:** {vehicle['cylinders']}
        - **Fuel:** {vehicle['fuel']}
        - **Mileage:** {vehicle['mileage']} miles
        - **Transmission:** {vehicle['transmission']}
        - **Trim:** {vehicle['trim']}
        - **Body:** {vehicle['body']}
        - **Doors:** {vehicle['doors']}
        - **Exterior Color:** {vehicle['exterior_color']}
        - **Interior Color:** {vehicle['interior_color']}
        - **Drivetrain:** {vehicle['drivetrain']}
        - **Description:** {vehicle['description']}
        """)
        st.write("---")

else:
    st.info("Provide your preferences above and click **Recommend Vehicles** to see recommendations.")
