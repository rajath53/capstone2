import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pandas as pd
import streamlit as st
oe = OrdinalEncoder()

#[theme]
backgroundColor="#2b10d6"
secondaryBackgroundColor="#2975ce"
textColor="#fafafb"


model = open('model.pkl','rb')
classifier = pickle.load(model)

#scale = open('scaler.pkl','rb')
#scaler = pickle.load(scale)

st.title("Telco_Customer_Churn")
st.header("Predicting Churned or not")

df = pd.read_csv("data.csv")
df = df.head()

nav = st.sidebar.radio("Navigation", ["Home","Prediction","About us"])
if nav == "Home":
    if st.checkbox("Show table"):
        st.table(df)

    val = st.slider("Filter df using MonthlyCharges",0,100)
    df=df.loc[df["MonthlyCharges"]>=val]
    
    graph = st.selectbox("What kind of graph ?", ["MonthlyCharges"])
    if graph == "MonthlyCharges":
        fig, ax = plt.subplots()
        ax.scatter(df['MonthlyCharges'],df['TotalCharges'])
        plt.title('Scatter')
        plt.ylim(0)
        plt.xlabel("MonthlyCharges")
        plt.ylabel("TotalCharges")
        st.pyplot(fig)

if nav == "Prediction":
    st.header("Know your Churn")
    gender = st.selectbox('Gender:', ['male', 'female'])
    SeniorCitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
    Partner= st.selectbox(' Customer has a partner:', ['yes', 'no'])
    Dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
    PhoneService = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
    MultipleLines = st.selectbox(' Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
    InternetService= st.selectbox(' Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
    OnlineSecurity= st.selectbox(' Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
    OnlineBackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
    DeviceProtection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
    TechSupport = st.selectbox(' Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
    StreamingTV = st.selectbox(' Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
    StreamingMovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
    Contract= st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
    PaperlessBilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
    PaymentMethod= st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check' ,'mailed_check'])
    tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
    MonthlyCharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
    TotalCharges = tenure*MonthlyCharges

    val = [[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]]
    val = np.array(val)
    X_test = pd.DataFrame(val)

    # Encoding the categorical data
    df_cat = X_test.select_dtypes("object")
    for i in df_cat:
        X_test[i]=oe.fit_transform(X_test[[i]])

    # Scaling the dataset
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)


    pred = classifier.predict(X_test)[0]
    if st.button("Predict"):
        if pred == 1:
            st.success("Your prediction is: Yes")
        elif pred == 0:
            st.success("Your prediction is: No")
        else:
            st.error("Invalid prediction value")