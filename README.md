# Disease Prediction from Medical Data 🫀

## 1. Project Title
Heart Disease Prediction using Machine Learning

## 2. Problem Statement
Heart disease is one of the leading causes of death globally. Early detection of heart disease can help healthcare professionals provide necessary interventions and improve patient outcomes. However, analyzing complex medical data manually can be challenging and error-prone.

## 3. Objective
To predict the possibility of heart disease based on patient medical data using machine learning classification algorithms. This project provides a comparative analysis of different models to find the best balance of recall and accuracy.

## 4. Dataset Information
The project uses the well-known **UCI Cleveland Heart Disease dataset**.
- **Instances:** 303
- **Attributes:** 14 (13 predictive features and 1 target)
- **Target:** Diagnosis of heart disease (1 = disease present, 0 = disease not present)

## 5. Features
1. **age:** Age in years
2. **sex:** Sex (1 = male, 0 = female)
3. **cp:** Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
4. **trestbps:** Resting blood pressure (in mm Hg)
5. **chol:** Serum cholestoral in mg/dl
6. **fbs:** Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg:** Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
8. **thalach:** Maximum heart rate achieved
9. **exang:** Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak:** ST depression induced by exercise relative to rest
11. **slope:** The slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
12. **ca:** Number of major vessels (0-3) colored by flourosopy
13. **thal:** Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect)

## 6. Technologies Used
- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib
- **Environment:** Jupyter Notebook
- **Deployment:** Streamlit

## 7. Machine Learning Algorithms
The following classification algorithms were implemented and compared:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

## 8. Data Preprocessing
- **Missing Value Handling:** Missing values in `ca` and `thal` were imputed using the mode to preserve dataset size and distribution.
- **Duplicate Removal:** Removed identical rows to prevent bias.
- **Feature Scaling:** `StandardScaler` was applied to numerical features for distance-based models (Logistic Regression, SVM) to ensure all features contribute equally.
- **Data Split:** 80-20 train-test split with stratification to maintain class balance.
- **Leakage Prevention:** Scalers were fit *only* on the training data and then used to transform the test data.

## 9. Exploratory Data Analysis (EDA)
- **Class Distribution:** The dataset is well-balanced (~54% disease, 46% no disease).
- **Correlations:** Features like chest pain (`cp`), max heart rate (`thalach`), and `slope` showed positive correlations with the target. Exercise-induced angina (`exang`) showed a negative correlation.
- **Outliers:** Outliers in blood pressure and cholesterol were retained as they represent valid, extreme medical conditions (e.g., severe hypertension).

## 10. Model Evaluation
In a medical context, **Recall (Sensitivity)** is prioritized.
A false negative (telling a sick patient they are healthy) is much more dangerous than a false positive (telling a healthy patient they might be sick, prompting further testing).
Metrics evaluated: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 11. Model Comparison
After training and hyperparameter tuning using `GridSearchCV`:
- **Logistic Regression** emerged as the final model due to an excellent balance of high Recall (~0.89) and strong ROC-AUC (~0.95), combined with its high interpretability in a clinical setting.
- Tree-based models like Random Forest and XGBoost performed well but tended to overfit slightly or yield lower recall compared to the tuned linear model.

## 12. How to Run the Project
1. Clone the repository or download the folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To view the analysis, open the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/disease_prediction.ipynb
   ```
4. To retrain the model from scratch:
   ```bash
   cd src
   python train.py
   ```

## 13. Streamlit Application Instructions
To run the interactive web app for predictions:
1. Navigate to the project root directory.
2. Run the Streamlit command:
   ```bash
   streamlit run app.py
   ```
3. Open the provided localhost URL in your browser.
4. Adjust the medical parameters using the UI controls and click **PREDICT**.

## 14. Project Limitations
- The dataset is relatively small (303 instances), which limits the model's ability to generalize to a broader population.
- The data is historical and may not reflect modern diagnostic criteria or demographics perfectly.

## 15. Future Improvements
- Collect a larger and more diverse dataset to improve generalization.
- Implement more advanced ensemble techniques or Deep Learning.
- Use SHAP (SHapley Additive exPlanations) values in the Streamlit app to explain *why* the model made a certain prediction for a specific patient.

## 16. Medical Disclaimer
⚠️ **Important:** This project is designed strictly for **educational and research purposes**. The machine learning model predictions should **not** be considered a medical diagnosis. Always consult a qualified healthcare professional for medical advice and clinical diagnosis.
