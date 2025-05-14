import requests
import json
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from plyer import notification

def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=10  # Duration in seconds
    )


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("C:/Users/niran/Downloads/updated_soil_pest_data.csv")  # Use the updated dataset

# Encode categorical features (State, Disease, Pest)
label_encoders = {}
for col in ["State", "Disease", "Likely Pest"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode risk levels (Disease & Pest)
risk_encoder = LabelEncoder()
df["Risk Level"] = risk_encoder.fit_transform(df["Risk Level"])
df["Pest Risk Level"] = risk_encoder.fit_transform(df["Pest Risk Level"])

# Train Disease Prediction Model
X_disease = df[["Temperature (°C)", "Soil Moisture (%)", "State"]]
y_disease = df["Disease"]
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)
disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(X_train_disease, y_train_disease)

# Train Pest Prediction Model
X_pest = df[["Temperature (°C)", "Soil Moisture (%)", "State"]]
y_pest = df["Likely Pest"]
X_train_pest, X_test_pest, y_train_pest, y_test_pest = train_test_split(X_pest, y_pest, test_size=0.2, random_state=42)
pest_model = RandomForestClassifier(n_estimators=100, random_state=42)
pest_model.fit(X_train_pest, y_train_pest)

# Train Risk Prediction Models (Disease & Pest)
X_risk = df[["Temperature (°C)", "Soil Moisture (%)", "State"]]
y_risk = df["Risk Level"]
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train_risk, y_train_risk)

y_pest_risk = df["Pest Risk Level"]
X_train_pest_risk, X_test_pest_risk, y_train_pest_risk, y_test_pest_risk = train_test_split(X_risk, y_pest_risk, test_size=0.2, random_state=42)
pest_risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
pest_risk_model.fit(X_train_pest_risk, y_train_pest_risk)

# Function to fetch soil data from API
def get_soil_data():
    url = "http://api.agromonitoring.com/agro/1.0/soil?polygon_id=67c08919dbbadd48e275b8d1&appid=1070ac6b2ac1acc66c1ef1538c2e72a9"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Extract temperature and moisture
        temp_kelvin = data.get("t0")  # Use "t0" for top layer temp
        moisture = data.get("moisture")

        if temp_kelvin is None or moisture is None:
            print("❌ Error: Missing values in API response")
            return None, None

        # Convert temperature from Kelvin to Celsius
        temp_celsius = temp_kelvin - 273.15
        print(f"✅ Debug: Parsed Data - Temperature: {temp_celsius:.2f}°C, Moisture: {moisture*100:.2f}%")

        return temp_celsius, moisture * 100  # Convert moisture to percentage

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        return None, None
    except json.JSONDecodeError:
        print("❌ Error: Failed to parse JSON response")
        return None, None

# Function to predict disease, risk level, pests, and crop at risk
def predict_for_karnataka():
    state = "Karnataka"  # Fixed state name

    # Fetch soil data from API
    temp, moisture = get_soil_data()

    if temp is None or moisture is None:
        print("❌ Prediction failed: Unable to fetch soil data")
        return

    # Ensure state exists in the dataset
    state_names = [s.lower().strip() for s in label_encoders["State"].classes_]
    if state.lower().strip() not in state_names:
        print(f"❌ Error: State '{state}' not found in dataset")
        return

    # Encode state
    state_encoded = label_encoders["State"].transform([state])[0]
    input_data = [[temp, moisture, state_encoded]]

    # Predict disease
    disease_encoded = disease_model.predict(input_data)[0]
    disease_pred = label_encoders["Disease"].inverse_transform([disease_encoded])[0]

    # Predict risk level (Disease)
    risk_encoded = risk_model.predict(input_data)[0]
    risk_label = risk_encoder.inverse_transform([risk_encoded])[0]

    # Find crop associated with the predicted disease
    crop_data = df[df["Disease"] == disease_encoded]["Crop"].dropna().unique()
    crop_at_risk = crop_data[0] if len(crop_data) > 0 else "Unknown Crop"

    # Predict pest
    pest_encoded = pest_model.predict(input_data)[0]
    pest_pred = label_encoders["Likely Pest"].inverse_transform([pest_encoded])[0]

    # Predict risk level (Pest)
    pest_risk_encoded = pest_risk_model.predict(input_data)[0]
    pest_risk_label = risk_encoder.inverse_transform([pest_risk_encoded])[0]

    print(f"✅ Prediction Success!")
    print(f"🌱 Predicted Disease: {disease_pred}")
    print(f"⚠️ Predicted Risk Level: {risk_label}")
    print(f"🌾 Crop at Risk: {crop_at_risk}")
    print(f"🦗 Likely Pest: {pest_pred}")
    print(f"⚠️ Pest Risk Level: {pest_risk_label}")

# Run the prediction for Karnataka
predict_for_karnataka()
