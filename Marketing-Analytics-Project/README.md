# 📊 Marketing Analytics: Amazon Review Sentiment & Customer Segmentation  (R)

## 🚀 Project Overview  
This project applies **text mining, sentiment analysis, and clustering** to analyze robotic vacuum cleaner reviews and segment customers for a **subscription model**. Using **Amazon reviews** and **customer transaction data**, we extract insights to help businesses optimize marketing strategies.

## 📊 Dataset Details  
- **Roomba Reviews Dataset** (`Roomba_Reviews.csv`)  
  - Contains **1,833** customer reviews of iRobot Roomba 650 & 880.  
- **Amazon Competitor Reviews Dataset** (`B06X1F3HXG_Reviews.csv`)  
  - Scraped **1,000+ reviews** of **iLife V5s Pro** from Amazon.  
  - Includes **title, star rating, review text, helpful votes.**  
- **Roomba Subscription Model Dataset** (`Roomba_Subscriber_Model_Dataset.csv`)  
  - Contains **3,643** customer records with **Recency, Frequency, and Monetary (RFM) values** for segmentation.

## 🛠️ Methods & Models Used  
👉 **Web Scraping (Amazon Reviews):** Extracted reviews using `rvest` and `RCurl`.  
👉 **Sentiment Analysis:** Measured polarity of reviews using `sentimentr`.  
👉 **Time-Based Sentiment Trends:** Analyzed how customer sentiment changed over time.  
👉 **Word Clouds:** Identified frequently used words in **positive & negative reviews**.  
👉 **Text Analytics:** Performed **POS tagging, verb/noun extraction, and topic modeling**.  
👉 **Customer Segmentation:** Applied **K-Means & Hierarchical Clustering** to identify Roomba customer segments.  

## 🔥 Key Results & Insights  
📌 **Sentiment Analysis Trends:**
- **Overall, positive reviews correlate strongly with higher star ratings.**
- **However, 2-star reviews for iLife V5s had a lower sentiment than 1-star reviews**, indicating **greater disappointment at 2 stars.**
- **Sentiment scores fluctuated over time**, especially for iLife V5s.

📌 **Word Cloud Analysis:**
- The word **"price" appeared frequently in iLife V5s reviews**, reflecting a price-conscious customer base.
- The words **"pet" and "dog" were dominant in Roomba reviews**, showing alignment with their pet-friendly marketing.

📌 **Customer Segmentation Findings:**
- Identified **three major customer groups**:
  - **Loyal subscribers** (high frequency, high monetary value).
  - **New buyers** (recent purchases, low frequency).
  - **One-time buyers** (high recency, low frequency).

## 🔍 Business Insights  
📌 **Common Customer Complaints & Feature Requests:**
- **Roomba reviews frequently mentioned battery life and navigation issues.**
- **iLife V5s reviews highlighted weak mopping performance.**
- **Customers requested better room mapping features and longer battery life.**

## 📂 Project Structure  
```
📂 Marketing-Analytics-Project  
 ├─📝 README.md                             # Project documentation  
 ├─ Marketing_Analytics.Rmd                  # Main R Markdown analysis   
 ├─📂 data                                   # Datasets  
 │   ├─ Roomba_Reviews.csv                   # Roomba product reviews  
 │   ├─ B06X1F3HXG_Reviews.csv               # Scraped Amazon competitor reviews  
 │   └─ Roomba_Subscriber_Model_Dataset.csv  # Customer segmentation data   
 └─📂 reports                                # Final reports & insights
     ├─ report.pdf                           # R Markdown PDF report
     └─ report.mhtml                         # R Markdown HTML report
```

## 📌 Future Improvements  
🚀 **Apply topic modeling (LDA) to extract key themes from Amazon reviews**  
🚀 **Enhance clustering with additional customer attributes**  
🚀 **Develop an interactive dashboard for customer segmentation**  

## 📬 Connect with Me  
If you found this useful, feel free to ⭐ star the repo or reach out:  
📧 Email: zachzhang1992@gmail.com  
💼 LinkedIn: [Hanzhe Zhang](https://www.linkedin.com/in/hanzhezhang)  
👨‍💻 GitHub: [HanzheZhang822](https://hanzhezhang822.github.io)  
