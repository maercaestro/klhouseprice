import pandas as pd
import numpy as np
import streamlit as st
import json
import xgboost as xgb
import datetime as dt
from sklearn.preprocessing import LabelEncoder

reg=xgb.XGBRegressor()
reg.load_model('housereg.json')

st.title("KL House Price Prediction by Abu Huzaifah")
st.write("""Using xgBoost ML Model to Predict KL House Price""")
st.write("""P/S This is using dataset from 4 years, so don't believe in the value. Just a simple end-to-end ML Project""")

locations=(
    'KLCC', 'Damansara Heights', 'Dutamas', 'Bukit Jalil',
       'Taman Tun Dr Ismail', 'Sri Petaling', 'Bukit Tunku (Kenny Hills)',
       'Mont Kiara', 'Desa ParkCity', 'Bangsar South', 'Ampang Hilir',
       'Kepong', 'Jalan Klang Lama (Old Klang Road)', 'Sungai Besi',
       'KL City', 'KL Sentral', 'Taman Melawati', 'Setapak',
       'City Centre', 'Country Heights Damansara', 'Taman Desa', 'Sentul',
       'Cheras', 'Bangsar', 'Segambut', 'Batu Caves', 'Wangsa Maju',
       'Ampang', 'Sri Hartamas', 'Klcc', 'Bukit Kiara', 'Setiawangsa',
       'OUG', 'Bukit Bintang', 'Jalan Sultan Ismail', 'Chan Sow Lin',
       'Jalan Kuching', 'Bandar Menjalara', 'KL Eco City', 'Seputeh',
       'Sunway SPK', 'Pantai', 'ADIVA Desa ParkCity', 'Kuchai Lama',
       'Jalan Ipoh', 'Mid Valley City', 'Brickfields', 'Desa Pandan',
       'Keramat', 'Pandan Indah', 'Desa Petaling', 'Federal Hill',
       'Other', 'Pandan Perdana', 'Bandar Damai Perdana', 'Puchong',
       'Salak Selatan', 'SEMARAK', 'Titiwangsa', 'Damansara',
       'Bandar Tasik Selatan', 'Alam Damai', 'Bukit Ledang', 'Jinjang',
       'Gombak', 'The Mines Resort', 'Sri Kembangan', 'Taman Duta',
       'Happy Garden', 'Taman Yarl', 'Pandan Jaya', 'U-THANT',
       'Taman TAR', 'Sungai Penchala', 'TAMAN MELATI', 'Off Gasing Indah',
       'Landed Sd', 'Seri Kembangan', 'Petaling Jaya', 'Kuala Lumpur',
       'Taman Wangsa Permai', 'taman cheras perdana',
       'Santuari Park Pantai', 'Sri Damansara', 'Bukit  Persekutuan',
       'Wangsa Melawati', 'Taman Yarl OUG', 'Kota Damansara',
       'Taman Ibukota', 'Ukay Heights', 'SANTUARI PARK PANTAI',
       'Singapore', 'Bandar Sri Damansara', 'cyberjaya',
       'Sungai Long SL8', 'Gurney', 'taman connaught', 'Bukit Damansara',
       'Kemensah', 'Jalan U-Thant', 'Rawang', 'Solaris Dutamas',
       'duta Nusantara', 'Bandar Sri damansara', 'Casa Rimba', 'kepong',
       'Taman Sri Keramat', 'Canary Residence', 'Taming Jaya'
)

property_types=(
    'Serviced Residence', 'Bungalow', 'Condominium (Corner)',
       'Semi-detached House', '2-sty Terrace/Link House (EndLot)',
       'Apartment (Intermediate)',
       '2-sty Terrace/Link House (Intermediate)',
       'Bungalow (Intermediate)', 'Semi-detached House (Intermediate)',
       'Bungalow (Corner)', 'Serviced Residence (Intermediate)',
       'Condominium', 'Condominium (Intermediate)',
       'Condominium (EndLot)', 'Serviced Residence (Corner)',
       '3-sty Terrace/Link House (Intermediate)',
       'Serviced Residence (Duplex)', '2-sty Terrace/Link House',
       '2-sty Terrace/Link House (Corner)',
       '2.5-sty Terrace/Link House (Intermediate)',
       '3-sty Terrace/Link House (Corner)',
       '3-sty Terrace/Link House (EndLot)',
       '3.5-sty Terrace/Link House (Intermediate)',
       'Serviced Residence (Penthouse)', 'Condominium (Studio)',
       '1-sty Terrace/Link House (Intermediate)',
       '1.5-sty Terrace/Link House (EndLot)', 'Apartment',
       'Condominium (Duplex)', 'Serviced Residence (EndLot)',
       '4-sty Terrace/Link House',
       '4-sty Terrace/Link House (Intermediate)', 'Townhouse',
       'Semi-detached House (Corner)', 'Townhouse (Intermediate)',
       'Apartment (Corner)', '3-sty Terrace/Link House',
       'Residential Land', '2.5-sty Terrace/Link House',
       '1.5-sty Terrace/Link House (Intermediate)',
       'Semi-detached House (EndLot)',
       '4.5-sty Terrace/Link House (Intermediate)',
       'Condominium (Penthouse)', '2.5-sty Terrace/Link House (EndLot)',
       'Serviced Residence (SOHO)', '3.5-sty Terrace/Link House',
       '3.5-sty Terrace/Link House (Corner)', '1-sty Terrace/Link House',
       'Residential Land (Corner)', 'Townhouse (EndLot)',
       '1-sty Terrace/Link House (Corner)', 'Townhouse (Corner)',
       'Bungalow Land', '1-sty Terrace/Link House (EndLot)',
       'Flat (Intermediate)', '1.5-sty Terrace/Link House (Corner)',
       'Bungalow Land (Intermediate)',
       '3-sty Terrace/Link House (Duplex)', 'Bungalow (EndLot)',
       'Residential Land (Intermediate)', 'Serviced Residence (Studio)',
       '1.5-sty Terrace/Link House',
       '2.5-sty Terrace/Link House (Penthouse)',
       '2.5-sty Terrace/Link House (Corner)', 'Flat (Corner)',
       'Apartment (EndLot)', '4-sty Terrace/Link House (Corner)', 'Flat',
       'Condominium (SOHO)', 'Bungalow Land (Corner)',
       'Serviced Residence (Triplex)', 'Flat (EndLot)',
       '3.5-sty Terrace/Link House (EndLot)', 'Apartment (Duplex)',
       'Bungalow (Duplex)', 'Semi-detached House (Triplex)',
       'Condominium (Triplex)', 'Cluster House (Intermediate)',
       '2-sty Terrace/Link House (Penthouse)', 'Apartment (Penthouse)',
       '3-sty Terrace/Link House (Triplex)', 'Townhouse (Duplex)',
       'Residential Land (EndLot)',
       '4-sty Terrace/Link House (Penthouse)',
       'Semi-detached House (Duplex)', 'Apartment (Studio)',
       '4.5-sty Terrace/Link House',
       '2.5-sty Terrace/Link House (Triplex)', 'Flat (Penthouse)',
       '2-sty Terrace/Link House (Duplex)', 'Cluster House',
       '2.5-sty Terrace/Link House (Duplex)', 'Apartment (Triplex)',
       'Semi-detached House (SOHO)', 'Bungalow (Penthouse)',
       'Cluster House (Corner)', 'Bungalow Land (EndLot)',
       '4.5-sty Terrace/Link House (Corner)'
)
size_types=(
    'Built-up', 'Land area'
)

furnishings=(
    'Fully Furnished', 'Partly Furnished', 'Unfurnished', 'Unknown'
)

location=st.selectbox("Select your location",locations)
property_type=st.selectbox("Select property type",property_types)
room=st.number_input("Select your room quantity")
bathroom=st.number_input("Select your bathroom quantity")
car_park=st.number_input("Select wether your property have car parks or not")
size=st.number_input("Select your property size in square feet")
size_type=st.selectbox("Select your size type",size_types)
furnishing=st.selectbox("Select furnishing type",furnishings)

def label_encoding(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column]=df[column].astype(str)
            le=LabelEncoder()
            df[column]=le.fit_transform(df[column])   
    return df

X = pd.DataFrame({'Location': [location],
                  'Rooms': [room],
                  'Bathrooms': [bathroom],
                  'Car Parks': [car_park],
                  'Property Type': [property_type],
                  'Furnishing': [furnishing],
                  'SizeType': [size_type],
                  'SizeValue': [size]})
X = label_encoding(X)

def predict(X):
    predictions=reg.predict(X)

    return predictions

button = st.button('Predict House Price')

if button:
    st.write('Predicted ART:',predict(X)[0]*1000000)
