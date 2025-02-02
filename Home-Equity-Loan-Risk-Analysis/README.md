# 📊 Loan Default & Risk Prediction Using Machine Learning (R)  

## 🚀 Project Overview  
This project was conducted over 8 weeks as part of my studies at Trine University, focusing on loan default risk modeling using Machine Learning in R. Throughout this period, I explored data preprocessing, feature engineering, decision trees, ensemble models, and model evaluation** to develop a predictive risk assessment tool for home equity loans.  

## 📊 Dataset Details  
- **HMEQ_Loss.csv** → Raw dataset with **5,960 records & 14 variables** (loan info, credit risk, financial history).  
- **HMEQ_Scrubbed.csv** → Processed dataset with **missing values imputed, categorical variables encoded, and feature flags created**.  
- **HMEQ_Dictionary.txt** → Data dictionary explaining variables.  

## 🛠️ Methods & Models Used  
✔ **Data Preprocessing (R)**: Imputation, One-Hot Encoding, Feature Engineering (`dplyr`, `tidyverse`)  
✔ **Exploratory Data Analysis (EDA)**: Histograms, Box Plots, Outlier Detection (`ggplot2`, `skimr`)  
✔ **Machine Learning Models (R):**  
  - **Decision Trees (`rpart`)**  
  - **Random Forest (`randomForest`)**  
  - **Gradient Boosting (`gbm`)**  
  - **Logistic Regression (`glm`)**  
✔ **Model Validation:** AUC Score, RMSE, Cross-Validation (`caret`, `pROC`)  
✔ **Feature Importance Analysis:** Identified **top risk factors** influencing loan default  

## 🔥 Key Results & Insights  
📌 **Predicted loan default with 88% accuracy**, improving risk assessment.  
📌 **Enhanced model performance by 14% (AUC) using ensemble learning.**  
📌 **Ranked top risk factors (Debt-to-Income, Delinquency, Mortgage Due) to improve lending decisions.**  
📌 **Estimated expected financial loss per loan** using probability × severity models.  

## 🔍 Skills Demonstrated  
✅ **R (`rpart`, `randomForest`, `gbm`, `glm`, `caret`, `ggplot2`)**  
✅ **Machine Learning (Decision Trees, RF, GBM, Logistic Regression)**  
✅ **Feature Engineering & Data Cleaning (`dplyr`, `tidyverse`)**  
✅ **Risk Modeling & Financial Forecasting**  

## 🛠️ How to Run This Project  
1️⃣ Clone this repository:  
   ```bash
   git clone https://github.com/HanzheZhang822/HanzheZhang822.github.io.git
   cd HanzheZhang822.github.io/Home-Equity-Loan-Risk-Analysis
   ```  
2️⃣ Install dependencies (R packages):  
   ```r
   source("scripts/install_packages.R")
   ```  
3️⃣ Run the analysis:  
   ```r
   source("scripts/Home-Equity-Loan-Risk-Analysis.R")  # Main analysis script
   ```  

## 📂 Project Structure  
```
📂 hmeq-loan-risk  
 ├─📝 README.md                          # Project Documentation  
 ├─📂 data                               # Dataset Files  
 │   ├─ HMEQ_Loss.csv                     # Raw dataset  
 │   ├─ HMEQ_Scrubbed.csv                  # Processed dataset  
 │   └─ HMEQ_Dictionary.txt                # Data dictionary  
 └─📂 scripts                            # R scripts for data processing & modeling  
 │   ├─ Home-Equity-Loan-Risk-Analysis.R   # Main analysis script  
 │   └─ install_packages.R                 # Dependencies  
```  

## 📌 Future Improvements  
🚀 **Try additional models (XGBoost, LightGBM) for better predictive performance.**  
🚀 **Analyze economic factors (interest rates, inflation) to enhance predictions.**  
🚀 **Develop an interactive dashboard for loan risk assessment.**  

## 📬 Connect with Me  
If you found this useful, feel free to ⭐ star the repo or reach out:  
📧 Email: zachzhang1992@gmail.com  
💼 LinkedIn: [Hanzhe Zhang](https://www.linkedin.com/in/hanzhezhang)  
👨‍💻 GitHub: [HanzheZhang822](https://hanzhezhang822.github.io)  
