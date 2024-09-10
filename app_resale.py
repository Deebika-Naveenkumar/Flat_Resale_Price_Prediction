import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("D:/vs code/flat/mrt.csv")
mrt_location = pd.DataFrame(data)

st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    layout="wide"
)

# for side bars
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
    
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 81%)]")

    try:
        with st.form("form1"):
            flat_type_options = ['3 ROOM', '5 ROOM', '4 ROOM', 'EXECUTIVE', '2 ROOM','MULTI-GENERATION', '1 ROOM']
            flat_model_options = ['Improved', 'Adjoined flat', 'Model A', 'New Generation', 'Standard', 'Apartment', 'Maisonette', 'Simplified',
                     'Model A-Maisonette', '2-room', 'Premium Apartment', 'Improved-Maisonette', 'Model A2', 'Multi Generation', 'DBSS',
                     'Type S1', 'Type S2', '3Gen', 'Premium Apartment Loft', 'Terrace', 'Premium Maisonette']
            
            # New Data inputs from the user for predicting the resale price
            street_name = st.text_input("Street Name")
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date')

            try:
                storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

                # Check if storey_range input follows the correct format
                if storey_range and ' TO ' in storey_range:
                    split_list = storey_range.split(' TO ')
                    try:
                        float_list = [float(i) for i in split_list]
                        storey_median = statistics.median(float_list)
                    except ValueError as ve:
                        st.write("Error: Please enter numeric values for storey range.")
                        storey_median = None
                else:
                    st.write("Error: Please use the correct format 'Value1 TO Value2'.")
                    storey_median = None

            except Exception as e:
                st.write(f"An unexpected error occurred: {e}")

            flat_type = st.selectbox("Flat Type",flat_type_options,key=1)
            flat_model = st.selectbox("Flat Model",flat_model_options,key=2)

            # button to submit the form details
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                # Load all necessary files
                with open("D:/vs code/flat/flattype.pkl", 'rb') as file:
                    flattype_encoder = pickle.load(file)
                with open("D:/vs code/flat/flatmodel.pkl", 'rb') as file:
                    flatmodel_encoder = pickle.load(file)
                with open("D:/vs code/flat/scaler.pkl", 'rb') as file:
                    scaler_loaded = pickle.load(file)
                with open("D:/vs code/flat/model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)

                # Encoding categorical variables
                flat_type_encoded = flattype_encoder.transform([[flat_type]]).toarray()
                flat_model_encoded = flatmodel_encoder.transform([[flat_model]]).toarray()

                # getting the address by joining the block number and the street name
                address = block + " " + street_name
                query_address = address
                query_string = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={query_address}&returnGeom=Y&getAddrDetails=Y"
                resp = requests.get(query_string)

                # Using OpenMap API getting the latitude and longitude location of that address
                origin = []
                data_geo_location = json.loads(resp.content)
                if data_geo_location['found'] != 0:
                    latitude = data_geo_location['results'][0]['LATITUDE']
                    longitude = data_geo_location['results'][0]['LONGITUDE']
                    origin.append((latitude, longitude))

                # Appending the Latitudes and Longitudes of the MRT Stations
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                mrt_lat = mrt_location['latitude']
                mrt_long = mrt_location['longitude']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # getting distance to nearest MRT Stations (Mass Rapid Transit System)
                list_of_dist_mrt = []
                for destination in range(0, len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                shortest = (min(list_of_dist_mrt))
                min_dist_mrt = shortest
                list_of_dist_mrt.clear()

                # Getting distance from CDB (Central Business District)
                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # Calculating lease_remain_years and other values
                lease_remain_years = 99 - (2024 - lease_commence_date)
                new_sample = np.array([[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                
                # Combine encoded categorical data with numerical data
                new_sample = np.hstack([new_sample, flat_type_encoded, flat_model_encoded])
                
                # Apply the scaler
                new_sample_scaled = scaler_loaded.transform(new_sample)
                
                # Make the prediction
                new_pred = loaded_model.predict(new_sample_scaled)[0]
                st.write('## :green[Predicted resale price:] ', np.exp(new_pred))
    except Exception as e:
        st.write(f"{e}")

