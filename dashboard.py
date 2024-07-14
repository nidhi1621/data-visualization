#dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


#title
st.set_page_config(page_title='Data Analysis Dashboard', page_icon=':bar_chart:',layout='wide')

# header and footer
header_container = st.container()
with header_container:
    st.image('https://docs.streamlit.io/logo.svg', width=100)
    st.title("Dashboard")


# footer container
# footer_container = st.container()
# with footer_container:
#     st.write("Data Analysis. © 2023")

#sidebar
sidebar = st.sidebar

# File upload 
uploaded_file = sidebar.file_uploader("Upload CSV file", type=['csv','xlsx'])
column_names = []
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    string_data = uploaded_file.getvalue().decode("ISO-8859-1")  
    stringio = StringIO(string_data)
    dataframe = pd.read_csv(stringio)
    
    if st.sidebar.checkbox("Show data1"):
        st.dataframe(dataframe)
    column_names = list(dataframe.columns)
    if st.sidebar.checkbox('column names'):
        st.write(dataframe.dtypes)
    if st.sidebar.checkbox("Correlation seaborn"):
        st.write(sns.heatmap(dataframe.corr(), annot=True))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

uploaded_file_2 = st.sidebar.file_uploader("Upload another CSV file", type=['csv', 'xlsx'], key="file_uploader_2")
column_names_2 = []

if uploaded_file_2 is not None:
    bytes_data_2 = uploaded_file_2.getvalue()
    string_data_2 = uploaded_file_2.getvalue().decode("ISO-8859-1")
    stringio_2 = StringIO(string_data_2)
    dataframe_2 = pd.read_csv(stringio_2, encoding="utf-8")

    if st.sidebar.checkbox("Show data2"):
        st.dataframe(dataframe_2)

    column_names_2 = list(dataframe_2.columns)

    if st.sidebar.checkbox('Column names', key="check2_checkbox"):
        st.write(dataframe_2.dtypes)

    selected_column = st.sidebar.selectbox("Select a column to encode", column_names_2)
    if selected_column:
        # Apply label encoding to the selected column
        encoder = LabelEncoder()
        dataframe_2[selected_column] = encoder.fit_transform(dataframe_2[selected_column])

        # Get user input for custom labels
        new_labels = st.sidebar.text_input("Enter the new labels (separated by comma): ")
        new_labels = [label.strip() for label in new_labels.split(",")]

        # Map encoded values to the custom labels
        mapping = dict(zip(encoder.classes_, new_labels))
        dataframe_2[selected_column] = dataframe_2[selected_column].map(mapping)

        # Display the updated dataframe
        st.write(dataframe_2)


    if st.sidebar.checkbox("Correlation seaborn", key="corr2_checkbox"):
        st.write(sns.heatmap(dataframe_2.corr(), annot=True))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

# Column selection 
   # Sidebar 
    options_1 = sidebar.multiselect('Select columns for Graph 1', column_names, column_names[:2])
    options_2 = sidebar.multiselect('Select columns for Graph 2', column_names_2, column_names_2[:2])
    graph_type_1 = sidebar.selectbox('Select graph type for Graph 1', ('Scatter', 'Line', 'Bar', 'Histogram', 'Pie', 'Violin', 'Area', 'Box','Density'))
    graph_type_2 = sidebar.selectbox('Select graph type for Graph 2', ('Scatter', 'Line', 'Bar', 'Histogram', 'Pie', 'Violin', 'Area', 'Box','Density'))

# Generate graphs based on user selection


    fig_1, fig_2 = None, None
    if graph_type_1 == 'Scatter':
        fig_1 = px.scatter(dataframe, x=options_1[0], y=options_1[1])
    elif graph_type_1 == 'Bar':
        fig_1 = px.bar(dataframe, x=options_1[0], y=options_1[1])
    elif graph_type_1 == 'Line':
        fig_1 = px.line(dataframe, x=options_1[0], y=options_1[1])
    elif graph_type_1 == 'Histogram':
        fig_1 = px.histogram(dataframe, x=options_1[0])
    elif graph_type_1 == 'Pie':
        fig_1 = px.pie(dataframe, values=options_1[1], names=options_1[0])
    elif graph_type_1 == 'Violin':
        fig_1 = px.violin(dataframe, x=options_1[0], y=options_1[1])
    elif graph_type_1 == 'Area':
        fig_1 = px.area(dataframe, x=options_1[0], y=options_1[1])
    elif graph_type_1 == 'Box':
        fig_1 = px.box(dataframe, x=options_1[0], y=options_1[1])
    else:
        fig_1 = px.density_contour(dataframe, x=options_1[0], y=options_1[1])


#performance evaluation
    if graph_type_2 == 'Bar':

     # training and testing
        X_train, X_test, y_train, y_test = train_test_split(dataframe[options_1[0]], dataframe[options_1[1]], test_size=0.2, random_state=42)

# Reshape the features
        X_train_reshaped = X_train.values.reshape(-1, 1)
        X_test_reshaped = X_test.values.reshape(-1, 1)

# Reshape the labels
        y_train_reshaped = y_train.values.reshape(-1, 1)
        y_test_reshaped = y_test.values.reshape(-1, 1)

# Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

# Train SVM regression model
        model = svm.SVR()
        model.fit(X_train_reshaped, y_train_reshaped)

# Predict values using the trained model
        y_pred = model.predict(X_test_reshaped)
        y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
        st.subheader("Performance Evaluation Graph1")
        mse = mean_squared_error(y_test_reshaped, y_pred)
        mae = mean_absolute_error(y_test_reshaped, y_pred)
    
        st.write("Mean Squared Error:", round(mse, 2))#decimal value that using round and 2 for represents
        st.write("Mean Absolute Error:", round(mae, 2))
    
    if graph_type_2 == 'Scatter':
        fig_2 = px.scatter(dataframe_2, x=options_2[0], y=options_2[1])
    elif graph_type_2 == 'Bar':
        fig_2 = px.bar(dataframe_2, x=options_2[0], y=options_2[1])
    elif graph_type_2 == 'Line':
        fig_2 = px.line(dataframe_2, x=options_2[0], y=options_2[1])
    elif graph_type_2 == 'Histogram':
        fig_2 = px.histogram(dataframe_2, x=options_2[0])
    elif graph_type_2 == 'Pie':
        fig_2 = px.pie(dataframe_2, values=options_2[1], names=options_2[0])
    elif graph_type_2 == 'Violin':
        fig_2 = px.violin(dataframe_2, x=options_2[0], y=options_2[1])
    elif graph_type_2 == 'Area':
        fig_2 = px.area(dataframe_2, x=options_2[0], y=options_2[1])
    elif graph_type_1 == 'Box':
        fig_1 = px.box(dataframe_2, x=options_1[0], y=options_1[1])
    else:
        fig_2 = px.density_contour(dataframe_2, x=options_2[0], y=options_2[1])

#performance evaluation
    if graph_type_2 == 'Bar':
#  training and testing
        X_train, X_test, y_train, y_test = train_test_split(dataframe_2[options_2[0]], dataframe_2[options_2[1]], test_size=0.2, random_state=42)

# Reshape the features
        X_train_reshaped = X_train.values.reshape(-1, 1)
        X_test_reshaped = X_test.values.reshape(-1, 1)

# Reshape the labels
        y_train_reshaped = y_train.values.reshape(-1, 1)
        y_test_reshaped = y_test.values.reshape(-1, 1)
# Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

# Train SVM regression model
        model = svm.SVR()
        model.fit(X_train_reshaped, y_train_reshaped)

# Predict values using the trained model
        y_pred = model.predict(X_test_reshaped)
        y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
        st.subheader("Performance Evaluation Graph2")
        mse = mean_squared_error(y_test_reshaped, y_pred)
        mae = mean_absolute_error(y_test_reshaped, y_pred)

        st.write("Mean Squared Error:", round(mse, 2))
        st.write("Mean Absolute Error:", round(mae, 2))
  

    # Display graphs 
    content = st.container()
    with content:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_1)
            st.markdown(f"### Graph 1:\n- Shows the relation between the columns {', '.join(options_1)} and graph type is {graph_type_1}")
        with col2:
            st.plotly_chart(fig_2)
            st.markdown(f"### Graph 2:\n- Shows the relation between the columns {', '.join(options_2)} and graph type is {graph_type_2}")

    if st.button('workspace'):
        col1,col2=st.columns(2)
        with col1:
            st.write(fig_1, "analysis1.png") 
        with col2:
            st.write(fig_2, "analysis2.png")


    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 10px;
            }
            .css-1e0fku0 {
                border-radius: 10px;
                box-shadow: 5px 5px 5px #888888;
            }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("Data Analysis. © 2023")

    # selected_column = st.sidebar.selectbox("Select a column to encode", column_names_2)
    # if selected_column:
    # Apply label encoding to the selected column
    #     encoder = LabelEncoder()
    #     dataframe_2[selected_column] = encoder.fit_transform(dataframe_2[selected_column])