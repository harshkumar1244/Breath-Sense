import os, sys, datetime as dt, traceback
import numpy as np
import pandas as pd
import requests

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QFileDialog, QGridLayout, QScrollArea,
    QMessageBox, QComboBox, QDoubleSpinBox, QCompleter
)
from PySide6.QtGui import QFont, QColor

# --- 1. Logic Helpers (ML & API) ---
ML_AVAILABLE = True
try:
    from aqi_model import load_model, prepare_timeseries, iterative_forecast_24h, TrainConfig
except Exception:
    ML_AVAILABLE = False

# PM2.5 to AQI Conversion
def pm25_to_aqi(pm25):
    try:
        if pm25 is None or (isinstance(pm25, float) and np.isnan(pm25)): return None
        bp = [(0.0,12.0,0,50), (12.1,35.4,51,100), (35.5,55.4,101,150),
              (55.5,150.4,151,200), (150.5,250.4,201,300), (250.5,350.4,301,400),
              (350.5,500.4,401,500)]
        x = float(pm25)
        for c_lo,c_hi,a_lo,a_hi in bp:
            if c_lo<=x<=c_hi: return ((a_hi-a_lo)/(c_hi-c_lo))*(x-c_lo)+a_lo
        return 500.0
    except: return None

# --- NEW ROBUST HEALTH SCORE FORMULA ---
MASK_FACTOR = {"None": 1.0, "Cloth": 1.4, "Surgical": 2.0, "N95": 10.0}

def compute_his(aqi_val, br, oh, mask):
    """
    Calculates Health Score (0-100) using a decay curve.
    Formula: 100 / (1 + (Exposure / K)^1.5)
    This ensures the score never hits 0.0 instantly and scales logicially.
    """
    if aqi_val is None or pd.isna(aqi_val): return 0.0
    
    mf = MASK_FACTOR.get(mask, 1.0)
    try:
        # 1. Total Intake Exposure
        # Base = AQI * Breathing Rate * Hours
        exposure = (float(aqi_val) * float(br) * float(oh))
        
        # 2. Apply Mask Protection
        reduced_exposure = exposure / float(mf)
        
        # 3. Calculate Score 
        # K is the "half-health" constant. 3000 is calibrated so AQI 100 (unhealthy) ~ Score 50.
        K = 3000.0 
        score = 100.0 / (1.0 + (reduced_exposure / K) ** 1.5)
        
        return max(0.0, min(100.0, score))
    except: return 0.0

# API Helpers
def geocode_city(name):
    try:
        r=requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name":name,"count":1}, timeout=10)
        js=r.json()
        if js.get("results"):
            g=js["results"][0]
            return float(g["latitude"]), float(g["longitude"])
    except: return None

def fetch_openweather_24h_pm25(lat, lon, api_key):
    if not api_key: raise ValueError("API Key missing")
    url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
    r = requests.get(url, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=15)
    r.raise_for_status()
    js = r.json()
    rows = []
    from datetime import datetime, timezone
    for h in js.get("list", [])[:24]:
        ts = datetime.fromtimestamp(h["dt"], tz=timezone.utc).astimezone()
        pm25 = h.get("components", {}).get("pm2_5", 0)
        rows.append({"timestamp": ts, "pm2_5": pm25, "aqi_us": pm25_to_aqi(pm25)})
    return pd.DataFrame(rows)

# --- 2. Modern UI Components ---
class MetricCard(QFrame):
    def __init__(self, title, icon=""):
        super().__init__()
        self.setStyleSheet("QFrame { background: rgba(255, 255, 255, 0.08); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }")
        lay = QVBoxLayout(self)
        self.t = QLabel(f"{icon} {title}")
        self.t.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 14px; font-weight: bold;")
        self.v = QLabel("--")
        self.v.setStyleSheet("color: white; font-size: 42px; font-weight: 900;")
        self.s = QLabel("")
        self.s.setStyleSheet("color: #8b5cf6; font-size: 12px;")
        lay.addWidget(self.t); lay.addWidget(self.v); lay.addWidget(self.s)

    def update_val(self, val, sub):
        self.v.setText(str(val))
        self.s.setText(sub)

class HourTile(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setStyleSheet("QFrame { background: rgba(255, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); }")
        lay = QVBoxLayout(self)
        self.lab = QLabel(label); self.lab.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 11px;")
        self.aqi = QLabel("AQI --"); self.aqi.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.his = QLabel("Health: --"); self.his.setStyleSheet("color: #a5b4fc; font-size: 12px;")
        lay.addWidget(self.lab); lay.addWidget(self.aqi); lay.addWidget(self.his)

    def set_data(self, aqi_val, his_val):
        a_txt = "--" if (aqi_val is None or np.isnan(aqi_val)) else str(int(round(aqi_val)))
        h_txt = "--" if (his_val is None or np.isnan(his_val)) else f"{his_val:.1f}"
        
        # Color coding for AQI
        color = "white"
        if aqi_val is not None and not np.isnan(aqi_val):
            if aqi_val > 300: color = "#ef4444" # Red
            elif aqi_val > 200: color = "#f97316" # Orange
            elif aqi_val > 100: color = "#eab308" # Yellow
            else: color = "#22c55e" # Green
            
        self.aqi.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        self.aqi.setText(f"AQI {a_txt}")
        self.his.setText(f"Health: {h_txt}")

# --- 3. Main Application ---
class BreathSenseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None; self.cfg = None; self.hist = None
        self.aqi_ml = None; self.his_ml = None
        self.aqi_api = None; self.his_api = None
        self.api_ready = False
        
        self.city_list = [
            "Delhi","New Delhi","Mumbai","Pune","Bengaluru","Hyderabad",
            "Chennai","Noida","Kolkata","Ahmedabad","Surat","Jaipur",
            "Lucknow","Kanpur","Gurugram","Patna","Varanasi","Nagpur",
            "Indore","Bhopal","Thane","Meerut","Rajkot","Vadodara","Agra","Ghaziabad",
            "London", "New York", "Tokyo", "Paris"
        ]
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Breath Sense — Modern Dual Forecast")
        self.resize(1200, 900)
        self.setStyleSheet("QWidget { background: #0f172a; color: white; font-family: 'Segoe UI'; }")
        
        main_lay = QVBoxLayout(self); main_lay.setContentsMargins(30,30,30,30); main_lay.setSpacing(20)

        # Header
        header = QLabel("🌍 Breath Sense"); header.setStyleSheet("font-size: 32px; font-weight: 900;")
        main_lay.addWidget(header)

        # Control Bar
        ctrl = QFrame(); ctrl.setStyleSheet("background: rgba(255,255,255,0.03); border-radius: 15px;")
        bar = QHBoxLayout(ctrl)
        
        self.cityIn = QLineEdit(); self.cityIn.setPlaceholderText("Search City...")
        self.cityIn.setStyleSheet("background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;")
        comp = QCompleter(self.city_list); comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.cityIn.setCompleter(comp)

        self.apiIn = QLineEdit(); self.apiIn.setPlaceholderText("API Key"); self.apiIn.setEchoMode(QLineEdit.Password)
        self.apiIn.setStyleSheet("background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;")
        env_key = os.getenv("OPENWEATHER_API_KEY", "")
        if env_key: self.apiIn.setText(env_key)

        self.mode = QComboBox(); self.mode.addItems(["ML forecast", "API forecast"])
        self.mode.setStyleSheet("background: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;")
        self.mode.currentIndexChanged.connect(self.render_mode)

        btnPred = QPushButton("Predict"); btnPred.clicked.connect(self.predict)
        btnPred.setStyleSheet("background: #d946ef; font-weight: bold; padding: 10px 20px; border-radius: 8px;")

        btnLoad = QPushButton("Load ML"); btnLoad.clicked.connect(self.load_all)
        btnLoad.setStyleSheet("background: #06b6d4; font-weight: bold; padding: 10px 20px; border-radius: 8px;")

        btnSave = QPushButton("Save CSV"); btnSave.clicked.connect(self.save_csv)
        btnSave.setStyleSheet("background: #22c55e; font-weight: bold; padding: 10px 20px; border-radius: 8px;")

        bar.addWidget(self.cityIn, 2); bar.addWidget(self.apiIn, 2); bar.addWidget(self.mode, 1)
        bar.addWidget(btnPred); bar.addWidget(btnLoad); bar.addWidget(btnSave)
        main_lay.addWidget(ctrl)

        # Inputs
        input_row = QHBoxLayout()
        self.inBR = QDoubleSpinBox(); self.inBR.setValue(10.0); self.inBR.setRange(1, 50)
        self.inOH = QDoubleSpinBox(); self.inOH.setValue(3.0); self.inOH.setRange(0, 24)
        self.inMask = QComboBox(); self.inMask.addItems(list(MASK_FACTOR.keys())); self.inMask.setCurrentText("Surgical")
        
        for w, l in [(self.inBR, "Breathing (L/min)"), (self.inOH, "Outdoor Hrs"), (self.inMask, "Mask Type")]:
            v = QVBoxLayout(); lbl = QLabel(l); lbl.setStyleSheet("font-size: 11px; color: grey;")
            w.setStyleSheet("background: rgba(255,255,255,0.1); border-radius: 5px; padding: 5px;")
            v.addWidget(lbl); v.addWidget(w); input_row.addLayout(v)
        main_lay.addLayout(input_row)

        # Metrics
        met_lay = QHBoxLayout()
        self.cCur = MetricCard("Current AQI", "🎯"); self.cPeak = MetricCard("Peak (24h)", "📈")
        self.cAvg = MetricCard("Average", "📊"); self.cHIS = MetricCard("Health Score", "❤️")
        met_lay.addWidget(self.cCur); met_lay.addWidget(self.cPeak); met_lay.addWidget(self.cAvg); met_lay.addWidget(self.cHIS)
        main_lay.addLayout(met_lay)

        # Tiles
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("background: transparent; border: none;")
        grid_w = QWidget(); self.grid = QGridLayout(grid_w)
        self.tiles = [HourTile("Now" if i==0 else f"+{i}h") for i in range(24)]
        for i, t in enumerate(self.tiles): self.grid.addWidget(t, i//6, i%6)
        scroll.setWidget(grid_w)
        main_lay.addWidget(scroll, stretch=1)

        self.status = QLabel("Ready"); main_lay.addWidget(self.status)
        
        if not ML_AVAILABLE:
            self.mode.setCurrentText("API forecast")
            self.mode.model().item(0).setEnabled(False)
            self.status.setText("ML libraries missing. Mode disabled.")

    def msg(self, text, critical=False):
        self.status.setText(text)
        if critical: QMessageBox.critical(self, "Error", text)

    def load_all(self):
        # 1. Check ML
        if not ML_AVAILABLE:
            return self.msg("ML not available — add aqi_model.py", True)
        
        # 2. Auto-Load Pickle
        try:
            self.model, self.cfg = load_model("aqi_model.pkl")
        except Exception as e:
            return self.msg(f"Failed to load 'aqi_model.pkl': {e}", True)

        # 3. Ask for CSV
        path, _ = QFileDialog.getOpenFileName(self, "Pick the CSV you trained on", ".", "CSV (*.csv)")
        if not path:
            return self.msg("CSV not selected.", True)
        
        try:
            # --- FIX: Clean Columns Logic ---
            raw = pd.read_csv(path)
            # Remove any accidental spaces in headers (e.g., 'PM2.5 ' -> 'PM2.5')
            raw.columns = raw.columns.str.strip() 
            
            # Additional Check: If model expects 'Date' but CSV has 'From Date'
            # (Adjust this if your specific CSV has known weird names, but strip() usually fixes it)
            
            self.hist = prepare_timeseries(raw, self.cfg)
            self.msg("Model (pkl) + CSV loaded successfully ✔")

        except KeyError as ke:
            # This catches the exact error from your screenshot
            self.msg(f"Column Mismatch! The CSV is missing columns required by the model: {ke}", True)
        except Exception as e:
            self.msg(f"File Load Error: {e}", True)

    def predict(self):
        # 1. ML Prediction
        if ML_AVAILABLE and self.hist is not None:
            try:
                # Run forecast
                pm = iterative_forecast_24h(self.model, self.hist, self.cfg)
                
                # Check for None result
                if pm is None or (isinstance(pm, (pd.Series, pd.DataFrame)) and pm.empty):
                    raise ValueError("Model returned empty prediction. Check input CSV data range.")
                
                # Create Index
                idx = pd.date_range(dt.datetime.now().replace(minute=0,second=0,microsecond=0), periods=24, freq="h")
                self.aqi_ml = pm.apply(pm25_to_aqi)
                self.aqi_ml.index = idx
                
                # Calc HIS for ML
                br, oh, mask = self.inBR.value(), self.inOH.value(), self.inMask.currentText()
                self.his_ml = pd.Series([compute_his(a, br, oh, mask) for a in self.aqi_ml], index=idx)
            
            except KeyError as ke:
                 # Catch 'not in index' errors specifically
                 self.aqi_ml = None; self.his_ml = None
                 self.msg(f"ML Data Error: Columns missing. {ke}", True)
            except Exception as e:
                self.aqi_ml = None; self.his_ml = None
                self.msg(f"ML Forecast Error: {str(e)}", False)

        # 2. API Prediction
        city = self.cityIn.text().strip()
        key = self.apiIn.text().strip()
        self.api_ready = False
        self.aqi_api = None; self.his_api = None
        
        if city and key:
            pos = geocode_city(city)
            if pos:
                try:
                    df = fetch_openweather_24h_pm25(pos[0], pos[1], key)
                    self.aqi_api = pd.Series(df["aqi_us"].values, index=df["timestamp"])
                    br, oh, mask = self.inBR.value(), self.inOH.value(), self.inMask.currentText()
                    self.his_api = pd.Series([compute_his(a, br, oh, mask) for a in self.aqi_api], index=df["timestamp"])
                    self.api_ready = True
                except Exception as e: self.msg(f"API Error: {e}", True)
            else:
                self.msg("City not found via API", True)
        
        self.render_mode()

    def render_mode(self):
        is_ml = (self.mode.currentText() == "ML forecast")
        
        aqi_data = None
        his_data = None
        source_name = ""

        if is_ml and self.aqi_ml is not None:
            aqi_data = self.aqi_ml
            his_data = self.his_ml
            source_name = "ML"
        elif self.api_ready and self.aqi_api is not None:
            aqi_data = self.aqi_api
            his_data = self.his_api
            source_name = "API"
        
        # Fallback
        if aqi_data is None:
            if self.aqi_ml is not None: 
                aqi_data = self.aqi_ml; his_data = self.his_ml; source_name = "ML (Fallback)"
            elif self.api_ready: 
                aqi_data = self.aqi_api; his_data = self.his_api; source_name = "API (Fallback)"

        if aqi_data is not None:
            # Update Cards
            self.cCur.update_val(int(aqi_data.iloc[0]), f"Now ({source_name})")
            self.cPeak.update_val(int(aqi_data.max()), "24h Peak")
            self.cAvg.update_val(int(aqi_data.mean()), "24h Avg")
            
            # Formatted HIS Score
            avg_his = his_data.mean()
            self.cHIS.update_val(f"{avg_his:.1f}", "Health Score")
            
            # Update Tiles
            vals = aqi_data.values
            h_vals = his_data.values
            for i, t in enumerate(self.tiles):
                if i < len(vals): t.set_data(vals[i], h_vals[i])
            self.msg(f"Displaying {source_name} Forecast")
        else:
            self.status.setText("No data. Load ML or enter City + Key.")

    def save_csv(self):
        if self.aqi_ml is None and self.aqi_api is None:
            return self.msg("Nothing to save", True)

        df = pd.DataFrame()
        if self.aqi_ml is not None:
            df["ML_AQI"] = self.aqi_ml.values
            df["ML_HIS"] = self.his_ml.values
            df.index = self.aqi_ml.index
            
        if self.aqi_api is not None:
            if df.empty:
                df.index = self.aqi_api.index
                df["API_AQI"] = self.aqi_api.values
                df["API_HIS"] = self.his_api.values
            else:
                df["API_AQI"] = pd.Series(self.aqi_api.values, index=df.index)
                df["API_HIS"] = pd.Series(self.his_api.values, index=df.index)

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "forecast.csv", "CSV (*.csv)")
        if path:
            df.to_csv(path)
            self.msg(f"Saved to {os.path.basename(path)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BreathSenseApp()
    window.show()
    sys.exit(app.exec())