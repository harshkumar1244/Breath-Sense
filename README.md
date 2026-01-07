Breath Sense — AQI 24h Predictor & Health Impact Score
Breath Sense is a desktop-based Python application that predicts next 24-hour Air Quality Index (AQI) and computes a personalized Health Impact Score (HIS) using Machine Learning and real-time API data. It features a modern PySide6 GUI, supports CSV-based ML forecasts, and optional OpenWeather API-based forecasts.
Features
•	24-hour AQI Forecast (hourly)
•	Machine Learning–based prediction using trained .pkl model
•	Live AQI forecast via OpenWeather API (PM2.5)
•	 Personalized Health Impact Score (0–100) based on:
o	AQI exposure
o	Breathing rate
o	Outdoor exposure hours
o	Mask type (None / Cloth / Surgical / N95)
•	Interactive dashboard with:
o	Current AQI
o	Peak AQI
o	Average AQI
o	Health Score
•	Export predictions to CSV
•	Modern dark-themed GUI (PySide6)
📁 Project Structure
BreathSense/
│
├── aqi_gui3.py                  # Main GUI application
├── aqi_model.pkl                # Trained ML model (required for ML mode)
├── aqi_model.py                 # Model utilities (load, forecast, config)
├── AQI_24h_Predictor_and_HealthScore_UPGRADED_ML.ipynb
│                               # Model training & experimentation notebook
├── your_dataset.csv             # Historical AQI CSV used for training
├── requirements.txt             # (Optional) dependency list
└── README.md                    # This file

How It Works
ML Forecast Mode
•	Uses a trained ML model (aqi_model.pkl)
•	Requires the same CSV structure used during training
•	Predicts PM2.5 → AQI → Health Score for the next 24 hours
API Forecast Mode
•	Uses OpenWeather Air Pollution Forecast API
•	Fetches real-time PM2.5 forecast based on city location
•	Converts PM2.5 → AQI (US EPA standard)
•	Computes Health Impact Score
You can switch between ML Forecast and API Forecast from the UI.

Requirements
Make sure Python 3.9 or later is installed.
Required Python Libraries
pip install numpy pandas requests PySide6 scikit-learn joblib
All required libraries are already installed in the developer's system. Install them if running on a new machine.

 OpenWeather API Key (Optional but Recommended)
To enable API Forecast mode, get a free API key from OpenWeather:
•	https://openweathermap.org/api
Set API Key (Recommended Method)
Windows (PowerShell)
set OPENWEATHER_API_KEY "your_api_key_here"
Linux / macOS
export OPENWEATHER_API_KEY = your_api_key_here
Alternatively, you can paste the API key directly into the GUI input field.

 How to Run
Step 1: Clone Repository
git clone https://github.com/your-username/breath-sense.git
cd breath-sense
Step 2: Run the Application
Run this command “python aqi_gui3.py”  in your command prompt in the same folder.

Using ML Forecast Mode
1.	Click "Load ML"
2.	Ensure aqi_model.pkl exists in the same directory
3.	Select the CSV file used during training
o	Column names must match training data
o	Extra spaces in headers are auto-handled
4.	Click "Predict"

Using API Forecast Mode
1.	Enter City Name (autocomplete supported)
2.	Enter OpenWeather API Key
3.	Select API forecast from dropdown
4.	Click "Predict"



Health Impact Score (HIS)
•	Range: 0 to 100
•	Higher score = healthier air exposure
•	Uses a non-linear decay model to avoid sudden zero scores
Factors Considered
Factor	Description
AQI	Air Quality Index
Breathing Rate	Liters/minute
Outdoor Hours	Exposure duration
Mask Type	Exposure reduction factor

Export Results
Click "Save CSV" to export:
•	ML AQI & HIS
•	API AQI & HIS (if available)
•	Hourly predictions (24 rows)

🛠 Troubleshooting
Issue: ML mode disabled
•	Ensure aqi_model.py and required ML libraries exist
Issue: Column mismatch error
•	Use the same CSV format as training
•	Column names are case-sensitive (spaces are auto-trimmed)
Issue: API not working
•	Check API key validity
•	Ensure city name is valid
________________________________________
Notes
•	Designed for research, learning, and demonstration purposes
•	AQI values follow US EPA PM2.5 standards
•	Health Score is indicative, not medical advice

 License
This project is open-source. You are free to use, modify, and distribute it for educational and non-commercial purposes.
________________________________________
Author
Harsh Kumar
Computer Science | Data Analytics | Machine Learning

