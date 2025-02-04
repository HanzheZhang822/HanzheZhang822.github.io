# **COVID-19 US 2020: Data Processing & Trend Analysis in SAS**  

### 📌 Project Overview  
This project analyzes **800K+ rows of U.S. COVID-19 data from 2020** using **SAS**. It focuses on **data cleaning, transformation, and visualization** to track **daily new cases, death rates, and trends at the state level**.  

### 📜 Data Source  
- [COVID-19 in USA Dataset - Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-usa)  

### 📊 Key Features  
✅ **Data Cleaning & Preprocessing**  
- Imported and processed **800K+ rows** of COVID-19 case data from Kaggle (`us_counties_covid19_daily.csv`).  
- Handled **missing values**, removed inconsistencies, and corrected cumulative case anomalies.  

✅ **State-Level Trend Analysis**  
- Computed **daily new cases and deaths** using `LAG()` functions.  
- Aggregated data **monthly** to analyze **death rates and case trends**.  

✅ **Automated Reporting & Visualization**  
- Created **monthly case and death rate reports** using `PROC SQL` and `PROC REPORT`.  
- Built **trend visualizations** using `PROC SGPLOT` and `PROC SGPANEL`.  

### 📂 Project Structure  
```
/covid19-us-2020-sas
│   ├── us_counties_covid19_daily.csv  # Raw COVID-19 dataset  
│── script/                   # SAS script  
│   ├── Covid19-US-2020-SAS.sas  # Main SAS script  
│── report/                    # Generated reports & charts  
│── README.md                   # Project documentation  
```

### 🚀 How to Run the Code  
1️⃣ Clone this repository:  
```bash
git clone https://github.com/your-github-username/covid19-us-2020-sas.git
```
2️⃣ Download the dataset from Kaggle:  
- **[COVID-19 in USA Dataset](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-usa)**  
- Place `us_counties_covid19_daily.csv` inside the `data/` folder.  

3️⃣ Open **SAS** and run `scripts/Covid19-US-2020-SAS.sas`.  

### 📈 Example Visualization  
*(Include a sample chart or summary report screenshot here.)*  


## 📬 Connect with Me  
If you found this useful, feel free to ⭐ star the repo or reach out:  
📧 Email: zachzhang1992@gmail.com  
💼 LinkedIn: [Hanzhe Zhang](https://www.linkedin.com/in/hanzhezhang)  
👨‍💻 GitHub: [HanzheZhang822](https://hanzhezhang822.github.io)  
