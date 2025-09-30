import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Predictive Denials", layout="wide")

# Title
st.title("🏥 Healthcare Claims Denial Prediction")
st.markdown("---")

# Function to generate synthetic claims data
@st.cache_data
def generate_claims_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    data = {
        'claim_amount': np.random.lognormal(8, 1.5, n_samples),
        'patient_age': np.random.randint(18, 90, n_samples),
        'prior_claims': np.random.poisson(3, n_samples),
        'days_to_submit': np.random.randint(1, 180, n_samples),
        'procedure_code': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'provider_type': np.random.choice(['Hospital', 'Clinic', 'Specialist'], n_samples),
        'insurance_type': np.random.choice(['PPO', 'HMO', 'Medicare', 'Medicaid'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create denial outcome based on rules (with some randomness)
    denial_prob = (
        (df['claim_amount'] > 10000) * 0.3 +
        (df['days_to_submit'] > 90) * 0.25 +
        (df['prior_claims'] > 5) * 0.2 +
        (df['procedure_code'] == 'E') * 0.15 +
        np.random.random(n_samples) * 0.3
    )
    
    df['denied'] = (denial_prob > 0.5).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Prepare features
    X = df.copy()
    y = X.pop('denied')
    
    # Convert categorical variables to numeric
    X = pd.get_dummies(X, columns=['procedure_code', 'provider_type', 'insurance_type'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, X_train.columns, accuracy, y_test, y_pred, feature_importance

# Generate data
df = generate_claims_data(1000)

# Sidebar
st.sidebar.header("📊 Dataset Overview")
st.sidebar.metric("Total Claims", len(df))
st.sidebar.metric("Denied Claims", df['denied'].sum())
st.sidebar.metric("Denial Rate", f"{df['denied'].mean()*100:.1f}%")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Model Performance", "📈 Data Analysis", "🎯 Predict New Claim"])

# Tab 1: Model Performance
with tab1:
    st.header("Model Performance")
    
    # Train model
    model, feature_names, accuracy, y_test, y_pred, feature_importance = train_model(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy")
        st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Approved', 'Denied'],
            y=['Approved', 'Denied'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Feature Importance")
        fig_importance = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Features Contributing to Denials"
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)

# Tab 2: Data Analysis
with tab2:
    st.header("Claims Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Denial rate by insurance type
        denial_by_insurance = df.groupby('insurance_type')['denied'].mean().reset_index()
        denial_by_insurance['denied'] = denial_by_insurance['denied'] * 100
        
        fig1 = px.bar(
            denial_by_insurance,
            x='insurance_type',
            y='denied',
            title="Denial Rate by Insurance Type",
            labels={'denied': 'Denial Rate (%)', 'insurance_type': 'Insurance Type'},
            color='denied',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Claim amount distribution
        fig3 = px.histogram(
            df,
            x='claim_amount',
            color='denied',
            title="Claim Amount Distribution by Outcome",
            labels={'denied': 'Denied', 'claim_amount': 'Claim Amount ($)'},
            nbins=30,
            barmode='overlay'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Denial rate by provider type
        denial_by_provider = df.groupby('provider_type')['denied'].mean().reset_index()
        denial_by_provider['denied'] = denial_by_provider['denied'] * 100
        
        fig2 = px.bar(
            denial_by_provider,
            x='provider_type',
            y='denied',
            title="Denial Rate by Provider Type",
            labels={'denied': 'Denial Rate (%)', 'provider_type': 'Provider Type'},
            color='denied',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Days to submit vs denial
        fig4 = px.scatter(
            df.sample(200),
            x='days_to_submit',
            y='claim_amount',
            color='denied',
            title="Claim Amount vs Days to Submit",
            labels={'denied': 'Denied', 'claim_amount': 'Claim Amount ($)', 'days_to_submit': 'Days to Submit'}
        )
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Predict New Claim
with tab3:
    st.header("Predict Denial Risk for New Claim")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claim_amount = st.number_input("Claim Amount ($)", min_value=0, max_value=100000, value=5000, step=100)
        patient_age = st.slider("Patient Age", min_value=18, max_value=90, value=45)
        prior_claims = st.slider("Number of Prior Claims", min_value=0, max_value=20, value=3)
    
    with col2:
        days_to_submit = st.slider("Days to Submit Claim", min_value=1, max_value=180, value=30)
        procedure_code = st.selectbox("Procedure Code", ['A', 'B', 'C', 'D', 'E'])
        provider_type = st.selectbox("Provider Type", ['Hospital', 'Clinic', 'Specialist'])
        insurance_type = st.selectbox("Insurance Type", ['PPO', 'HMO', 'Medicare', 'Medicaid'])
    
    if st.button("🎯 Predict Denial Risk", type="primary"):
        # Create input dataframe
        new_claim = pd.DataFrame({
            'claim_amount': [claim_amount],
            'patient_age': [patient_age],
            'prior_claims': [prior_claims],
            'days_to_submit': [days_to_submit],
            'procedure_code': [procedure_code],
            'provider_type': [provider_type],
            'insurance_type': [insurance_type]
        })
        
        # Prepare for prediction
        new_claim_encoded = pd.get_dummies(new_claim, columns=['procedure_code', 'provider_type', 'insurance_type'])
        
        # Align with training features
        for col in feature_names:
            if col not in new_claim_encoded.columns:
                new_claim_encoded[col] = 0
        
        new_claim_encoded = new_claim_encoded[feature_names]
        
        # Predict
        prediction = model.predict(new_claim_encoded)[0]
        probability = model.predict_proba(new_claim_encoded)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("🚫 HIGH RISK - Likely to be Denied")
            else:
                st.success("✅ LOW RISK - Likely to be Approved")
        
        with col2:
            st.metric("Approval Probability", f"{probability[0]*100:.1f}%")
        
        with col3:
            st.metric("Denial Probability", f"{probability[1]*100:.1f}%")
        
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1]*100,
            title={'text': "Denial Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*This is a demonstration application using synthetic data for educational purposes.*")
