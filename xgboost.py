import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_rows', 1000)
from collections import Counter
from geopy.distance import great_circle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dropout, Dense, LSTM, Embedding, Conv1D, MaxPool1D, Flatten
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Sequential
import keras
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

cal = pd.read_csv('calendar.csv~0/calendar.csv')
lis = pd.read_csv('listings.csv~0/listings.csv')
rev = pd.read_csv('reviews.csv~0/reviews.csv')


columns_to_keep = ['id',  'description', 'host_has_profile_pic', 'neighbourhood_group_cleansed', 
                   'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',  
                   'bedrooms', 'bed_type', 'amenities', 'price', 'cleaning_fee', 
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  
                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy']



df = lis[columns_to_keep].set_index('id')
df.cleaning_fee.fillna('$0.00', inplace=True)
df.security_deposit.fillna('$0.00', inplace=True)
df.price = df.price.str.replace('$', '').str.replace(',', '').astype(float)
df.cleaning_fee = df.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)
df.security_deposit = df.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)
df.extra_people = df.extra_people.str.replace('$', '').str.replace(',', '').astype(float)

df.drop(df[ (df.price > 400) | (df.price == 0) ].index, axis=0, inplace=True)
df.dropna( inplace=True)
df.host_has_profile_pic.fillna(value='f', inplace=True)
def distance_to_mid(lat, lon):
    berlin_centre = (52.5027778, 13.404166666666667)
    accommodation = (lat, lon)
    return great_circle(berlin_centre, accommodation).km

df['distance'] = df.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)
df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
df['size'] = df['size'].str.replace("\D", "")
df['size'] = df['size'].astype(float)
df.drop(['description'], axis=1, inplace=True)


df['Laptop_friendly_workspace'] = df['amenities'].str.contains('Laptop friendly workspace')
df['TV'] = df['amenities'].str.contains('TV')
df['Family_kid_friendly'] = df['amenities'].str.contains('Family/kid friendly')
df['Host_greets_you'] = df['amenities'].str.contains('Host greets you')
df['Smoking_allowed'] = df['amenities'].str.contains('Smoking allowed')
df['Wifi'] = df['amenities'].str.contains('Wifi')
df['Kitchen'] = df['amenities'].str.contains('Kitchen')
#df['Heating'] = df['amenities'].str.contains('Heating')
df['Essentials'] = df['amenities'].str.contains('Essentials')
df['Hair_dryer'] = df['amenities'].str.contains('Hair dryer')
#df['Cable TV'] = df['amenities'].str.contains('Cable TV')
df['Bed_linens'] = df['amenities'].str.contains('Bed linens')
#df['Shampoo'] = df['amenities'].str.contains('Shampoo')
df['Internet'] = df['amenities'].str.contains('Internet')
df['Elevator'] = df['amenities'].str.contains('Elevator')
df['Refrigerator'] = df['amenities'].str.contains('Refrigerator')
df['Dishes_and_silverware'] = df['amenities'].str.contains('Dishes and silverware')
#df['Hot_water'] = df['amenities'].str.contains('Hot water')
df['Stove'] = df['amenities'].str.contains('Stove')
#df['Smoking_allowed'] = df['amenities'].str.contains('Smoking allowed')


one_hot = pd.get_dummies(df['host_has_profile_pic'], prefix = 'host_has_profile_pic')
df.drop("host_has_profile_pic",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['neighbourhood_group_cleansed'], prefix = 'neighbourhood_group_cleansed')
df.drop("neighbourhood_group_cleansed",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['property_type'], prefix = 'property_type')
df.drop("property_type",inplace = True, axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['room_type'], prefix = 'room_type')
df.drop("room_type",inplace = True, axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['bed_type'], prefix = 'bed_type')
df.drop("bed_type",inplace = True, axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['instant_bookable'], prefix = 'instant_bookable')
df.drop("instant_bookable",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['is_business_travel_ready'], prefix = 'is_business_travel_ready')
df.drop("is_business_travel_ready",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['cancellation_policy'], prefix = 'cancellation_policy')
df.drop("cancellation_policy",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Laptop_friendly_workspace'], prefix = 'Laptop_friendly_workspace')
df.drop("Laptop_friendly_workspace",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['TV'], prefix = 'TV')
df.drop("TV",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Family_kid_friendly'], prefix = 'Family_kid_friendly')
df.drop("Family_kid_friendly",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Host_greets_you'], prefix = 'Host_greets_you')
df.drop("Host_greets_you",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Smoking_allowed'], prefix = 'Smoking_allowed')
df.drop("Smoking_allowed",inplace = True, axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['Wifi'], prefix = 'Wifi')
df.drop("Wifi",inplace = True, axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['Kitchen'], prefix = 'Kitchen')
df.drop("Kitchen",inplace = True, axis = 1)
df = df.join(one_hot)


# one_hot = pd.get_dummies(df['Heating'], prefix = 'Heating')
# df.drop("Heating",inplace = True, axis = 1)
# df = df.join(one_hot)


one_hot = pd.get_dummies(df['Essentials'], prefix = 'Essentials')
df.drop("Essentials",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Hair_dryer'], prefix = 'Hair_dryer')
df.drop("Hair_dryer",inplace = True, axis = 1)
df = df.join(one_hot)

# one_hot = pd.get_dummies(df['Cable TV'], prefix = 'Cable_TV')
# df.drop("Cable TV",inplace = True, axis = 1)
# df = df.join(one_hot)

one_hot = pd.get_dummies(df['Bed_linens'], prefix = 'Bed_linens')
df.drop("Bed_linens",inplace = True, axis = 1)
df = df.join(one_hot)

# one_hot = pd.get_dummies(df['Shampoo'], prefix = 'Shampoo')
# df.drop("Shampoo",inplace = True, axis = 1)
# df = df.join(one_hot)


one_hot = pd.get_dummies(df['Internet'], prefix = 'Internet')
df.drop("Internet",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Elevator'], prefix = 'Elevator')
df.drop("Elevator",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Refrigerator'], prefix = 'Refrigerator')
df.drop("Refrigerator",inplace = True, axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Dishes_and_silverware'], prefix = 'Dishes_and_silverware')
df.drop("Dishes_and_silverware",inplace = True, axis = 1)
df = df.join(one_hot)

# one_hot = pd.get_dummies(df['Hot_water'], prefix = 'Hot_water')
# df.drop("Hot_water",inplace = True, axis = 1)
# df = df.join(one_hot)

one_hot = pd.get_dummies(df['Stove'], prefix = 'Stove')
df.drop("Stove",inplace = True, axis = 1)
df = df.join(one_hot)


df.drop(['latitude', 'longitude', 'amenities', 'size'], axis=1, inplace=True)
y = df['price'].values
X = df
X.drop(['price'],axis = 1, inplace=True)
sc = MinMaxScaler()
sc_x = sc.fit(X)
X = sc_x.transform(X)
sc = MinMaxScaler()
sc_y = sc.fit(y.reshape(-1,1))
y = sc_y.transform(y.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=21)




pca = PCA(n_components = 64)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
booster = XGBRegressor()
booster.fit(X_train, y_train)
y_pred = booster.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))
y_t = sc_y.inverse_transform(y_test)


plt.figure(figsize=(25,15))
plt.plot(y_pred, label ='y_pred')
plt.plot(y_t, label = 'real')
plt.xlabel('features_row number')
plt.ylabel('price')
plt.title('Using Xgboost')
plt.legend()
plt.savefig('xgboost.png')


