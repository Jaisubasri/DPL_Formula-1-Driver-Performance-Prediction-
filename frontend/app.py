import streamlit as st
from streamlit_option_menu import option_menu
import webbrowser
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import os
import tempfile
import sys

# Set up Streamlit app with multipage navigation
st.set_page_config(page_title="Time Series App", layout="wide")
# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Preprocessing", "Feature Engineering", "📈 Model & Predictions", " 📊 Visualization"],
        icons=["gear", "filter","cut-line","cut-line"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Preprocessing":
    st.title("📂 Data Preprocessing")
    # Display HTML files
    st.write("### View Reports")
    data_dir = ("/mount/src/dpl_formula-1-driver-performance-prediction-/data_statistics")
    html_files = [f for f in os.listdir(data_dir) if f.endswith(".html")]
    st.markdown("<h3 style='text-align: center;'> STATISTICS OF THE PREPROCESSED FILES </h3>", unsafe_allow_html=True)
    cols = st.columns(4) 

    for idx, file in enumerate(html_files):
        display_name = '_'.join(file.rsplit('_', 2)[:-2])
        button_style = f"""
            <style>
                div.stButton > button {{
                    width: 100%;
                    height: 60px;
                    border-radius: 10px;
                    background-color: black;
                    color: white;
                    font-size: 14px;
                }}
                div.stButton > button:hover {{
                    background-color: red;
                    color: black;
                }}
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        with cols[idx % 4]:  # Arrange in 4 columns
            if st.button(display_name):
                webbrowser.open(f'file:////mount/src/dpl_formula-1-driver-performance-prediction-/data_statistics/{file}')
                
elif selected == "Feature Engineering":
    
    def load_data():
        return pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/feature_engineering/Normalized_engineered_features.csv")  # Replace with your actual CSV path

    def feature_engineering():
        st.title("🛠 Feature Engineering")

        # Load data
        df = load_data()

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head())

        # Feature Importance (Using random weights for demo)
        feature_importance = {
            "DriverConsistency": 40,
            "TeamStrength": 35,
            "TrackComplexity": 25
        }

        # Bar Chart for Feature Importance
        st.subheader("📊 Feature Importance")
        fig = px.bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            labels={'x': 'Features', 'y': 'Importance (%)'},
            title="Feature Importance Chart",
            color=list(feature_importance.values()),
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig)

        # Feature Correlation Heatmap
        st.subheader("📈 Feature Correlation Heatmap")
        corr = df[["DriverConsistency", "TeamStrength", "TrackComplexity"]].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Feature Distribution
        st.subheader("📊 Feature Distributions")
        for feature in ["DriverConsistency", "TeamStrength", "TrackComplexity"]:
            fig = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
            st.plotly_chart(fig)

        # Interactive Filtering
        st.subheader("🔗 Filter Data")
        driver_id = st.selectbox("Select Driver ID", df['driverId'].unique())
        constructor_id = st.selectbox("Select Constructor ID", df['constructorId'].unique())
        circuit_id = st.selectbox("Select Circuit ID", df['circuitId'].unique())

        filtered_df = df[(df['driverId'] == driver_id) &
                        (df['constructorId'] == constructor_id) &
                        (df['circuitId'] == circuit_id)]

        st.subheader("📋 Filtered Data")
        st.dataframe(filtered_df)
        
    feature_engineering()

elif selected=="📈 Model & Predictions":
    
    st.title("🏎️ F1 Driver Standings Predictor")
    
    def prepare_input_data(race_data, results_data, drivers_data, constructors_data, status_data):
        
        race_data.rename(columns={'name': 'event', 'date': 'EventDate'}, inplace=True)
        merged_data = results_data.merge(race_data[['raceId', 'year', 'event', 'EventDate']], on='raceId', how='left')

        
        drivers_data['FullName'] = drivers_data['forename'] + " " + drivers_data['surname']
        merged_data = merged_data.merge(drivers_data[['driverId', 'FullName', 'nationality']], on='driverId', how='left')

       
        constructors_data.rename(columns={'name': 'TeamName'}, inplace=True)
        merged_data = merged_data.merge(constructors_data[['constructorId', 'TeamName', 'nationality']], on='constructorId', how='left', suffixes=('_driver', '_team'))

        
        merged_data = merged_data.merge(status_data[['statusId', 'status']], on='statusId', how='left')

        # Feature Engineering
        merged_data['Status_Finished'] = merged_data['status'].apply(lambda x: 1 if 'Finished' in str(x) else 0)
        merged_data['is_podium'] = merged_data['positionOrder'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        merged_data['is_first'] = merged_data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
        merged_data['is_top_two'] = merged_data['positionOrder'].apply(lambda x: 1 if x in [1, 2] else 0)
        merged_data['is_winner'] = merged_data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

        # Handling Missing Values
        merged_data.replace('\\N', np.nan, inplace=True)
        merged_data = merged_data.apply(pd.to_numeric, errors='ignore')
        merged_data.fillna(0, inplace=True)

        # Convert EventDate to datetime
        merged_data['EventDate'] = pd.to_datetime(merged_data['EventDate'], errors='coerce')

        # Encode Categorical Variables
        label_encoders = {}
        categorical_columns = ['FullName', 'nationality_driver', 'TeamName', 'nationality_team', 'status']

        for col in categorical_columns:
            le = LabelEncoder()
            merged_data[col] = le.fit_transform(merged_data[col].astype(str))
            label_encoders[col] = le  # Save encoders for inverse transformation if needed

        # Scale Numerical Features
        scaler = MinMaxScaler()
        numerical_columns = ['grid', 'positionOrder', 'points', 'laps', 'milliseconds']
        merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

        return merged_data, label_encoders, scaler
    
    drivers_data=pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/preprocessed_dataset/drivers_preprocessed.csv" )
    constructors_data=pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/preprocessed_dataset/constructors_preprocessed.csv")
    race_data=pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/preprocessed_dataset/races_preprocessed.csv")
    results_data=pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/preprocessed_dataset/results_preprocessed.csv")
    status_data=pd.read_csv("/mount/src/dpl_formula-1-driver-performance-prediction-/preprocessed_dataset/status_preprocessed.csv")
    prepared_data, label_encoders, scaler = prepare_input_data(race_data, results_data, drivers_data, constructors_data, status_data)

    model_file = model = load_model("/mount/src/dpl_formula-1-driver-performance-prediction-/model/lstm_time_series_model.keras")

    if model_file:
            # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
            tmp_file.write(model_file.read())
            tmp_model_path = tmp_file.name

        # Load the .keras model with custom_objects
        model = load_model(tmp_model_path, custom_objects={'Orthogonal': Orthogonal})
        st.success("✅ Model Loaded Successfully!")

        st.write("### Preview of Data", prepared_data.head())

        # Preprocess Data
        driver_data = prepared_data.groupby(['year', 'driverId'])['points'].sum().reset_index()
        driver_data = driver_data.pivot(index='year', columns='driverId', values='points').fillna(0)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(driver_data)

        # Create Sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        SEQ_LENGTH = 3
        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        # Model Evaluation
        loss = model.evaluate(X, y, verbose=0)
        st.metric(label="📉 Model Loss (MSE)", value=f"{loss:.4f}")
        st.metric(label="📉 Accuracy", value=f"{1-loss:.4f}")

        # Prediction for Next Year
        last_sequence = scaled_data[-SEQ_LENGTH:]
        last_sequence = last_sequence.reshape((1, SEQ_LENGTH, X.shape[2]))

        predicted_scaled = model.predict(last_sequence)
        predicted_points = scaler.inverse_transform(predicted_scaled)[0]

        driver_ids = driver_data.columns
        predicted_2025 = dict(zip(driver_ids, predicted_points))

        # Display Predictions
        st.write("### 🏆 Predicted Driver Standings for 2025")
        sorted_drivers = sorted(predicted_2025.items(), key=lambda x: x[1], reverse=True)
        for rank, (driver, points) in enumerate(sorted_drivers[0:10], 1):
            st.write(f"{rank}. **Driver {driver}** - {points:.1f} points")

        # Visualization
        best_driver_index = np.argmax(predicted_points)
        best_driver_2025 = driver_data.columns[best_driver_index]

        st.write(f"### 📈 Predicted Best Driver for 2025: **Driver {best_driver_2025}**")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(driver_data.index, driver_data[best_driver_2025], label=f"Driver {best_driver_2025}", linewidth=2)
        ax.plot(driver_data.index[-1] + 1, predicted_points[best_driver_index], 'ro', label='2025 Prediction')
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Points")
        ax.set_title(f"Performance Trend for Driver {best_driver_2025}")
        ax.legend()
        st.pyplot(fig)


        # Visualization 3: Multi-Driver Performance
        st.write("### 📊 Driver Performance Over Time")
        selected_drivers = st.multiselect("Select Drivers", driver_ids, default=driver_ids[:3])
        fig, ax = plt.subplots()
        for driver in selected_drivers:
            ax.plot(driver_data.index, driver_data[driver], label=f"Driver {driver}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Points")
        ax.set_title("Driver Performance Comparison")
        ax.legend()
        st.pyplot(fig)

        # Visualization 4: Year-over-Year Improvement
        st.write("### 📈 Year-over-Year Points Improvement")
        yoy_improvement = driver_data.diff().fillna(0)
        fig, ax = plt.subplots()
        for driver in selected_drivers:
            ax.plot(yoy_improvement.index, yoy_improvement[driver], label=f"Driver {driver}")
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel("Year")
        ax.set_ylabel("Points Change")
        ax.set_title("Year-over-Year Points Improvement")
        ax.legend()
        st.pyplot(fig)