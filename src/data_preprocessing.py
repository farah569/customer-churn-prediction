import pandas as pd
import numpy as np 

def preprocess_data():
    #1-load dataset
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(),inplace=True)
    df.drop("customerID",axis=1,inplace=True)
    df["Churn"]=df["Churn"].map({"Yes": 1, "No": 0})
    df=pd.get_dummies(df,drop_first=True)
    df.to_csv("data/processed/clean_data.csv", index=False)
    print("Clean data saved to data/processed/clean_data.csv")


    


if __name__ =="__main__":
    preprocess_data()
