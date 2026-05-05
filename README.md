# 🚀 Experimentation & Uplift Modeling: Salvaging a "Failed" A/B Test

**Live Web App:** https://uplift-modeling-portfolio-project-xsappcht8nbthqfjkbqlzem.streamlit.app/

## 📌 The Business Problem
Standard A/B testing measures the **Average Treatment Effect (ATE)**. However, aggregate metrics can hide crucial insights. In this project, an initial marketing campaign showed a **statistically significant negative lift (-1.18%)**, meaning the campaign seemingly failed. 

Standard practice would dictate scrapping the campaign. However, applying Causal Inference and Machine Learning reveals a different story: **Heterogeneous Treatment Effects**. 

## 🧠 The Solution: Uplift Modeling (T-Learner)
Instead of looking at the average, I built a Meta-Learner (T-Learner using Random Forests) to predict the **individual treatment effect** for every user. 

This model segments customers into four distinct quadrants:
1. **Persuadables:** Customers who only buy *because* of the campaign. (Target!)
2. **Sleeping Dogs:** Customers who are annoyed by the campaign and will churn if contacted. (Do NOT target!)
3. **Sure Things:** Customers who will buy regardless. (Don't waste marketing spend on them.)
4. **Lost Causes:** Customers who will never buy. (Don't waste marketing spend on them.)

**The Insight:** The campaign failed on average because it severely alienated the "Sleeping Dogs," which dragged down the massive gains from the "Persuadables." By deploying this model, we can filter out the Sleeping Dogs and turn a net-negative campaign into a highly profitable targeting strategy.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science & ML:** `scikit-learn` (Random Forest), `statsmodels` (Frequentist A/B Testing), `pandas`, `numpy`
* **Frontend/Deployment:** `Streamlit`

## 💻 How to Run Locally
1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Run the interactive web app: `streamlit run app.py`

## 📂 Project Structure
* `notebooks/uplift_exploration.ipynb`: The raw statistical analysis, power calculation, and T-Learner model training.
* `app.py`: The productionized Streamlit application.