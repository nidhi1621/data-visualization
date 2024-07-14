#confusion matrix.py
# Train a classification model on your data
import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#sidebar
sidebar = st.sidebar

#upload file
uploaded_file = sidebar.file_uploader("Upload CSV file", type=['csv','xlsx'])
column_names = []
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    string_data = uploaded_file.getvalue().decode("ISO-8859-1")  
    stringio = StringIO(string_data)
    dataframe = pd.read_csv(stringio)
    if st.sidebar.checkbox("Display Data"):
        st.dataframe(dataframe)
    column_names = list(dataframe.columns)

    #select column to analyze
    target_column = st.sidebar.selectbox('Select target column', dataframe.columns)
    X = dataframe.drop(target_column, axis=1)
    y = dataframe.iloc[:, dataframe.columns.get_loc(target_column)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression()#logistic regression method for probability
    clf.fit(X_train, y_train)#fit method

    # Make predictions on the test set  
    y_pred = clf.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Get label names from target column
    label_names = dataframe[target_column].unique()

    # Plot the confusion matrix on heatmap
    fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=label_names, y=label_names)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig)

#Train and Test dataset also split that Train/Test is a method to measure the accuracy of your model.
#Split the data into 80% and 20% (Pareto Principle) the 80 represents the 80% of the time that data scientists expend getting data ready for use and the 20 refers to the mere 20% of their time that goes into actual analysis and reporting.
#80% of the data in the training set, 10% in the validation set, and 10% in the test set