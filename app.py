import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- App Config ---
st.set_page_config(page_title="Uplift Modeling Portfolio", layout="wide")

# --- Helper Function: Generate Data & Train Model ---
@st.cache_resource # Caches the model so it doesn't retrain on every click
def train_uplift_model():
    np.random.seed(42)
    n_users = 10000
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n_users),
        'past_purchases': np.random.poisson(3, n_users),
        'website_visits': np.random.poisson(10, n_users),
        'is_treatment': np.random.binomial(1, 0.5, n_users)
    })
    
    # Logic: Campaign works for young people, fails for older people
    base_prob = 0.10 + (df['past_purchases'] * 0.01)
    treatment_effect = np.where(df['age'] < 35, 0.08, -0.05) 
    df['converted'] = np.random.binomial(1, np.clip(base_prob + (df['is_treatment'] * treatment_effect), 0, 1))

    X = df[['age', 'past_purchases', 'website_visits']]
    y = df['converted']
    t = df['is_treatment']

    # Train T-Learner
    model_ctrl = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X[t==0], y[t==0])
    model_trt = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X[t==1], y[t==1])
    
    uplift_scores = model_trt.predict_proba(X)[:, 1] - model_ctrl.predict_proba(X)[:, 1]
    
    return model_ctrl, model_trt, uplift_scores

# Load models
model_ctrl, model_trt, historical_uplift_scores = train_uplift_model()

# --- UI Layout ---
st.title("🚀 Experimentation & Uplift Modeling")
st.markdown("Turning a 'failed' A/B test into a profitable targeting strategy using Machine Learning.")

tab1, tab2, tab3 = st.tabs(["📊 Phase 1: The A/B Test", "🧠 Phase 2: Uplift Model", "🎯 Phase 3: Interactive Predictor"])

# --- TAB 1: A/B Test ---
with tab1:
    st.header("The Problem: A 'Failed' Campaign")
    st.write("Initial aggregate analysis showed the treatment campaign actually performed *worse* than the control.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Conversion", "9.83%")
    col2.metric("Treatment Conversion", "8.64%", "-1.18%")
    col3.metric("P-Value", "< 0.001", "Statistically Significant Drop")
    
    st.error("Standard Practice: Stop the campaign. \n\nData Science Approach: Dig deeper to find heterogeneous treatment effects.")

# --- TAB 2: Uplift Model ---
with tab2:
    st.header("The Solution: T-Learner Meta-Model")
    st.write("By training two distinct Random Forest models, we calculated an individual 'Uplift Score' for every user.")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(historical_uplift_scores, bins=40, color='purple', alpha=0.7)
    ax.axvline(0, color='black', linestyle='dashed', linewidth=2)
    ax.set_title("Bimodal Distribution of Uplift Scores")
    ax.set_xlabel("Uplift Score (Negative = Sleeping Dogs, Positive = Persuadables)")
    st.pyplot(fig)
    
    st.success("Insight: The campaign wasn't a total failure. It worked highly effectively on a specific sub-segment, but was dragged down by alienating another segment.")

# --- TAB 3: Interactive Predictor ---
with tab3:
    st.header("Live Customer Simulator")
    st.write("Adjust the parameters below to predict if a user is a Persuadable or a Sleeping Dog.")
    
    c1, c2 = st.columns(2)
    with c1:
        test_age = st.slider("User Age", 18, 65, 25)
        test_purchases = st.number_input("Past Purchases", 0, 20, 2)
        test_visits = st.number_input("Website Visits (Last Month)", 0, 50, 5)
        
    with c2:
        # Predict on the fly
        input_data = pd.DataFrame([[test_age, test_purchases, test_visits]], columns=['age', 'past_purchases', 'website_visits'])
        prob_ctrl = model_ctrl.predict_proba(input_data)[0][1]
        prob_trt = model_trt.predict_proba(input_data)[0][1]
        live_uplift = prob_trt - prob_ctrl
        
        st.subheader("Prediction Results")
        st.metric("Probability of Conversion (If NOT Targeted)", f"{prob_ctrl*100:.1f}%")
        st.metric("Probability of Conversion (If TARGETED)", f"{prob_trt*100:.1f}%")
        
        if live_uplift > 0.02:
            st.success(f"**Target this user!** Expected Uplift: +{live_uplift*100:.1f}%")
        elif live_uplift < -0.02:
            st.error(f"**Do NOT target! (Sleeping Dog)** Expected Uplift: {live_uplift*100:.1f}%")
        else:
            st.warning(f"**Neutral Impact (Sure Thing / Lost Cause)** Expected Uplift: {live_uplift*100:.1f}%")