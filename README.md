# Short-Term Photovoltaic Power Prediction & Microgrid EMS using LSTM

This repository contains a complete end-to-end implementation of a **Long Short-Term Memory (LSTM)** neural network designed to predict short-term photovoltaic (PV) power generation in a smart microgrid environment. The project addresses the inherent uncertainty of solar energy production to optimize energy storage management, minimize main grid dependency, and reduce operational costs.

## 🎯 Project Purpose

Solar power generation is highly intermittent due to weather fluctuations (e.g., clouds, temperature changes). This project tackles this uncertainty by:

* **Predicting Future PV Production:** Feeding historical meteorological time-series data into an LSTM neural network to forecast solar generation for the coming hours.
* **Smart Microgrid Energy Management (EMS):** Implementing a rule-based energy management system that utilizes PV forecasts to intelligently manage battery charging/discharging and grid energy exchange.
* **System Robustness & Adaptability:** Simulating real-world challenges including sensor noise, weather forecast errors, and online learning (incremental model retraining).
* **Baseline Comparison:** Benchmarking the LSTM model against a traditional persistence model to prove its superiority.

## 📊 Dataset & Preprocessing

The dataset is obtained from the **Photovoltaic Geographical Information System (PVGIS)** of the European Union, specifically tailored for **Thessaloniki, Greece** (coordinates metadata: slope 35°, azimuth 2°, installed peak PV power 1kWp, system loss 14%, crystalline silicon technology).

* **Size:** ~166,000 hourly measurements spanning from 2005 to 2023.
* **Features Used:**
  * **Power ($P$):** PV Power output in Watts (Target variable, max ~930W for a 1kWp system).
  * **Radiation:** Direct ($Gb(i)$), Diffuse ($Gd(i)$), and Reflected ($Gr(i)$) in $W/m^2$.
  * **Sun Height ($H\_sun$):** Solar elevation angle in degrees.
  * **Temperature ($T2m$):** Ambient temperature at 2 meters (°C).
  * **Wind Velocity ($WS10m$):** Wind speed at 10 meters ($m/s$).
* **Data Cleaning & Engineering:**
  * Filtered out nighttime data ($Sun\_Height \le 0$), reducing the working dataset to ~80,800 daytime samples.
  * Normalized all features to $[0, 1]$ using `MinMaxScaler` (fitted exclusively on the training split to prevent data leakage).
  * Created rolling time-windows (24 hours history) to structure the input sequences for the LSTM.
  * Chronologically split the data into **Training (80%)** and **Testing (20%)**.

## 🧠 System Architecture & Methodology

### 1. LSTM Neural Network Model
* **Architecture:** Sequential model comprising an LSTM layer (64 units) for capturing temporal dependencies, a Dropout layer ($p=0.2$) for overfitting prevention, and a Dense output layer predicting the next power value.
* **Optimization:** Trained using the Adam optimizer and Mean Squared Error (MSE) loss function for 20 epochs with mini-batches of 64.

### 2. Microgrid Energy Management System (EMS)
A rule-based controller that operates hourly based on predicted PV output versus household load (modeled dynamically between 150W and 500W):
1. **Surplus Energy ($PV > Load$):** Excess energy charges the battery up to its maximum capacity limit ($1800Wh$). Any further surplus is exported (sold) to the main grid.
2. **Deficit Energy ($PV < Load$):** Battery discharges to cover the deficit down to a safe minimum limit ($200Wh$). If the battery is depleted, power is imported (bought) from the main grid.

### 3. Real-World Robustness Simulations
* **Sensor Noise Test:** Injects Gaussian noise ($\pm5\%$) into test inputs to evaluate prediction stability.
* **Weather Forecast Uncertainty:** Simulates imperfect meteorological forecasts by adding higher error factors ($\pm15\%$) specifically to weather features.
* **Online Learning:** Implements incremental model updates (`train_on_batch` / chunked retraining) over time to adapt to seasonal shifts and environmental changes.

## 🛠️ Technologies & Libraries

* **Deep Learning:** TensorFlow / Keras (`Sequential`, `LSTM`, `Dense`, `Dropout`).
* **Data Manipulation & Analysis:** `pandas`, `numpy`.
* **Preprocessing & Evaluation:** Scikit-learn (`MinMaxScaler`, `mean_squared_error`, `mean_absolute_error`).
* **Visualization:** `matplotlib`.
* **Environment:** Google Colab.

## 🏆 Evaluation & Key Findings

* **Accuracy:** The LSTM model achieved an impressive **MAE of ~47.83 W** and **RMSE of ~84.13 W** on the test set, corresponding to roughly ~5% deviation from peak capacity.
* **Baseline Comparison:** Outperformed the naive persistence model by **57% in MAE** and **43% in RMSE**, proving that deep learning is vital for accurate solar forecasting.
* **Microgrid Self-Sufficiency:** The EMS simulation demonstrated that accurate PV forecasting enables high system efficiency, achieving a remarkable **90.5% self-sufficiency** rate over a simulated test week.
