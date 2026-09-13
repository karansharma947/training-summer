# Superstore Profit & Loss Predictor — Streamlit App

A 4-page dashboard built with Streamlit:

1. **Profit/Loss Predictor** (home) — enter order details, get an instant Profit/Loss prediction with confidence.
2. **Data View** — filter and browse the cleaned dataset, with a download button.
3. **Visualizations** — profit/sales breakdowns by category, region, discount, and time.
4. **Findings & Conclusion** — model comparison table (Logistic Regression vs Decision Tree vs Random Forest) and auto-computed business insights.

## How to run

1. Make sure `app.py` and `superstore_cleaned.csv` are in the **same folder**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. It will open automatically at `http://localhost:8501`.

## Notes

- The three models (Logistic Regression, Decision Tree, Random Forest) are trained **live**, right when the app starts, using the same logic as `Superstore_Prediction_model.py`. The best one (by Loss F1-Score) is auto-selected for predictions on the Home page — no separate `.pkl` file needed.
- Training is cached (`st.cache_resource`), so it only happens once per app session, not on every click.
- If you'd rather load a pre-trained `.pkl` bundle instead of training on startup, that's a quick swap — just ask.
