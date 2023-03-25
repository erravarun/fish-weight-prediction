import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
data = pd.read_csv("C:\\Users\\Varun Kumar\\Desktop\\fish-weight-prediction\\fish.csv")
# print(data.head())
# print(data.info)
# print(data.isnull().sum())
# print(data['Weight'].nunique())
# print(data['Species'].value_counts())
# sns.countplot('Species',data = data,palette = 'hls')
# plt.show()
import plotly.express as px
fig = px.histogram(data,x = 'Species',color = 'Species')
# fig.show()
# sns.pairplot(data)
# plt.show()
# print(data.corr())
plt.figure(figsize=(15,6))
# sns.heatmap(data.corr(),annot = True)
# plt.show()
# sns.boxplot(data['Weight'])
plt.xticks(rotation = 90)
# plt.show()
fish_weight = data['Weight']                                        #--identifying outliers
Q3 = fish_weight.quantile(.75)
Q1 = fish_weight.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
weight_outliers = fish_weight[(fish_weight <lower_limit) | (fish_weight> upper_limit)]
# print(weight_outliers)
sns.boxplot(data['Length1'])
plt.xticks(rotation = 90)
# plt.show()
fish_Length1 = data['Length1']
Q3 = fish_Length1.quantile(.75)
Q1 = fish_Length1.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
length1_outliers = fish_Length1[(fish_Length1 <lower_limit) | (fish_Length1> upper_limit)]
# print(length1_outliers)
fish_Length2 = data['Length2']
Q3 = fish_Length2.quantile(.75)
Q1 = fish_Length2.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
length2_outliers = fish_Length2[(fish_Length2 <lower_limit) | (fish_Length2> upper_limit)]
# print(length2_outliers)
fish_Length3 = data['Length3']
Q3 = fish_Length3.quantile(.75)
Q1 = fish_Length3.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
length3_outliers = fish_Length3[(fish_Length3 <lower_limit) | (fish_Length3> upper_limit)]
# print(length3_outliers)
fish_Height = data['Height']
Q3 = fish_Height.quantile(.75)
Q1 = fish_Height.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
Height_outliers = fish_Height[(fish_Height <lower_limit) | (fish_Height > upper_limit)]
# print(Height_outliers)
fish_Width = data['Width']
Q3 = fish_Width.quantile(.75)
Q1 = fish_Width.quantile(.25)
IQR = Q3 - Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
Width_outliers = fish_Height[(fish_Width <lower_limit) | (fish_Width > upper_limit)]
# print(Width_outliers)
# print(data[142:145])
fish_data_new  = data.drop([142,143,144])                   #--removing outliers
# print(fish_data_new)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaling_columns = ['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']
fish_data_new[scaling_columns] = scaler.fit_transform(fish_data_new[scaling_columns])
# print(fish_data_new.describe())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
fish_data_new['Species'] = label_encoder.fit_transform(fish_data_new['Species'].values)
X = fish_data_new.drop('Weight',axis = 1)
y = fish_data_new['Weight']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LogisticRegression
import xgboost as xgb
model1 = RandomForestRegressor()
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
print("Training Accuracy1:",model1.score(X_train,y_train))
print("Testing Accuracy1:",model1.score(X_test,y_test))
model2 = LinearRegression()
model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
print("Training Accuracy2:",model2.score(X_train,y_train))
print("Testing Accuracy2:",model2.score(X_test,y_test))
model3 = DecisionTreeRegressor()
model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)
print("Training Accuracy3:",model3.score(X_train,y_train))
print("Testing Accuracy3:",model3.score(X_test,y_test))
# model4 = LogisticRegression()                 #--error we cannot use logistic regression
# model4.fit(X_train,y_train)
# y_pred = model4.predict(X_test)
# print("Training Accuracy4:",model4.score(X_train,y_train))
# print("Testing Accuracy4:",model4.score(X_test,y_test))
xgb1 = xgb.XGBRegressor() 
xgb1.fit(X_train,y_train)
xgb_pred = xgb1.predict(X_test)
print("Training Accuracyxgb:",xgb1.score(X_train,y_train))
print("Testing Accuracyxgb:",xgb1.score(X_test,y_test))
xgb1.save_model("model.json")
import streamlit as st
st.header("Fish Weight Prediction App")
st.text_input("Enter your Name:",key = "name")
np.save('classes.npy',label_encoder.classes_)
label_encoder.classes_ = np.load('classes.npy',allow_pickle=True)
xgb_best = xgb.XGBRegressor()
xgb_best.load_model("model.json")
if st.checkbox('Show Training DataFrame'):
    fish_data_new
st.subheader('Please select relevant features of your Fish')
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio('Name of the Fish:',np.unique(data['Species']))
input_length1 = st.slider('Vertical Length(cm)',0.0,max(data['Length1']),1.0)
input_length2 = st.slider('Diagonal Length(cm)',0.0,max(data['Length2']),1.0)
input_length3 = st.slider('Cross Length(cm)',0.0,max(data['Length3']),1.0)
input_Height = st.slider('Height(cm)',0.0,max(data['Height']),1.0)
input_Width = st.slider('Diagonal Width(cm)',0.0,max(data['Width']),1.0)
if st.button('Make Prediction'):
    input_species = label_encoder.transform(np.expand_dims(inp_species,-1))
    inputs = np.expand_dims([int(input_species),input_length1,input_length2,input_length3,input_Height,input_Width],0)
    prediction = xgb_best.predict(inputs)
    print("Final Pred",np.squeeze(prediction,-1))
    st.write(f"Your Fish Weight is:{np.squeeze(prediction,-1):.2f}g")

