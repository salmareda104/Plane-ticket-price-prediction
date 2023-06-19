import streamlit as st
import pandas as pd
import joblib

Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

def prediction(Airline, Source, Destination, Dep_Hour, Arrival_Hour, Duration_min, Stops, Additional_Info, Journey_Month_num, Journey_Day_num):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0, "Airline"] = Airline
    test_df.at[0, "Source"] = Source
    test_df.at[0, "Destination"] = Destination
    test_df.at[0, "Dep_Hour"] = Dep_Hour
    test_df.at[0, "Arrival_Hour"] = Arrival_Hour
    test_df.at[0, "Duration_min"] = Duration_min
    test_df.at[0, "Stops"] = Stops
    test_df.at[0, "Additional_Info"] = Additional_Info
    test_df.at[0, "Journey_Month_num"] = Journey_Month_num
    test_df.at[0, "Journey_Day_num"] = Journey_Day_num
    st.dataframe(test_df)
    result = Model.predict(test_df)[0]
    return result

def main():
    st.title("Flight Ticket Price Prediction")
    Airline = st.selectbox("Airline", ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 'Vistara',
                                       'Air Asia', 'GoAir', 'Multiple carriers Premium economy',
                                       'Jet Airways Business', 'Vistara Premium economy', 'Trujet'])
    Source = st.selectbox("Source", ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai'])
    Destination = st.selectbox("Destination", ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])
    Dep_Hour = st.slider("Dep_Hour", min_value=0, max_value=23, value=0, step=1)
    Arrival_Hour = st.slider("Arrival_Hour", min_value=0, max_value=23, value=0, step=1)
    Duration_min = st.slider("Duration_min", min_value=0, max_value=50, value=0, step=1)
    Stops = st.selectbox("Stops", [0, 1, 2, 3])
    Additional_Info = st.selectbox("Additional_Info", ['no info', 'in-flight meal not included',
                                                       'no check-in baggage included', '1 long layover',
                                                       'change airports', 'business class', '1 short layover',
                                                       'red-eye flight', '2 long layover'])
    Journey_Month_num = st.selectbox("Journey_Month_num", [3, 4, 5, 6])
    Journey_Day = st.selectbox("Journey_Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Journey_Day_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}[Journey_Day]
    
    if st.button("Predict Flight Cost"):
        result = prediction(Airline, Source, Destination, Dep_Hour, Arrival_Hour, Duration_min, Stops, Additional_Info, Journey_Month_num, Journey_Day_num)
        st.text(f"The flight ticket will cost {result} dollars")

if __name__ == '__main__':
    main()
