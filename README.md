# Algerian Forest Fire Risk Prediction

## Overview

This project is an end-to-end machine learning web application that predicts the **Fire Weather Index (FWI)** for Algerian forest regions using meteorological and fire-related indicators.

The system takes real-world environmental inputs and returns:
- a **numeric Fire Weather Index (FWI)**
- a **clear fire risk level** (Low / Moderate / High / Extreme)

The application is built using **Flask** and a **scikit-learn regression model**, and is designed to be deployed in a production environment.

---

## What is Fire Weather Index (FWI)?

The **Fire Weather Index (FWI)** is a standard metric used to estimate wildfire danger based on weather and fuel conditions.

General interpretation:
- **FWI < 10** → Low risk  
- **FWI 10–20** → Moderate risk  
- **FWI 20–30** → High risk  
- **FWI > 30** → Extreme risk  

Higher values indicate faster fire spread and increased difficulty of control.

---

## Input Features

The model uses the following **9 input features**:

1. Temperature (°C)  
2. Relative Humidity (%)  
3. Wind Speed (km/h)  
4. Rain (mm)  
5. FFMC (Fine Fuel Moisture Code)  
6. DMC (Duff Moisture Code)  
7. ISI (Initial Spread Index)  
8. Classes (Fire / No Fire indicator)  
9. Region (Bejaia / Sidi Bel-Abbes)

**Target Variable:**  
- Fire Weather Index (FWI)

---

## Model Details

- **Algorithm:** Ridge Regression  
- **Preprocessing:** StandardScaler  
- **Library:** scikit-learn  

The trained model and scaler are saved using pickle and reused during inference to ensure consistency between training and deployment.

---

