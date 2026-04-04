# 🏗️ Muraqib | مراقب 

**Muraqib** is an AI-powered project delay predictor designed for the Saudi construction industry. It uses machine learning (Random Forest) to analyze various project risk factors and predict the probability of activity delays.

## 🚀 Features
- **Real-time Prediction**: Input activity details (Complexity, Supply, Weather, etc.) to get an instant risk assessment.
- **Bilingual Interface**: Full support for Arabic and English.
- **Visual Analytics**: Interactive dashboards for historical project performance and risk factor analysis.
- **Data-Driven Logic**: Uses a realistic, stochastic risk scoring engine with built-in uncertainty (noise).
- **Risk-Sensitive Balancing**: Tuned to prioritize the detection of potential delays (High Recall).

## 📊 Current Model Performance (v8 - Balanced)
- **Recall (Risk Sensitivity)**: **96.12%** (Exceptional at catching potential delays) 
- **ROC-AUC**: **95.87%**
- **Accuracy**: ~83%
- **Key Predictor**: Subcontractor Performance (~44.5% importance)

## 📁 Project Structure
- `app.py`: Streamlit dashboard and UI.
- `src/muraqib/model.py`: Random Forest training and prediction logic (now with balanced weighting).
- `src/muraqib/data_loader.py`: Data ingestion and realistic risk enrichment (stochastic logic).
- `src/muraqib/i18n.py`: Internationalization (AR/EN) support.
- `evaluate_model.py`: Performance evaluation script.
- `eval_results.txt`: Latest detailed evaluation report.

## 🔧 Installation & Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Run evaluation:
   ```bash
   python evaluate_model.py
   ```

---
*Powered by Scikit-Learn | Developed for Construction Risk Analytics*