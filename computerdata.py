# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Computer_Data =pd.read_csv(r"C:\Users\ANKIT\Desktop\Aradhana\Multilinear\Computer_Data.csv")

# to get top 6 rows
Computer_Data.head() # to get top n rows use Computer_Data.head(10)
cleanup={"cd":{"yes":"yup","no":"nope"},"multi":{"yes":"y","no":"n"}}
Computer_Data.replace(cleanup,inplace=True)
Computer_Data.head()

Computer_Data=Computer_Data[['price','speed','hd','ram','screen','cd','multi','premium','ads','trend']]
inputvariables=list(Computer_Data)
del inputvariables[0]
inputvariables

outputvariables=list(Computer_Data)[0]
outputvariables

inputdata=Computer_Data[inputvariables]
inputdata

catcolumns=['cd','multi','premium']
for column in catcolumns:
    dummyCols=pd.get_dummies(inputdata[column]) # for each column in catcolumns its creating dummy values
    inputdata=inputdata.join(dummyCols)  #after that it joining the dummy columns to the input datasets
    del inputdata[column] #after the original column for which dummy variables is created will be declared here
inputdata
outputdata=Computer_Data[[outputvariables]]

import statsmodels.formula.api as smf
import statsmodels.api as sm

inputdata=sm.add_constant(inputdata)
model=sm.OLS(outputdata,inputdata).fit()
model.summary()

sm.graphics.influence_plot(model)

inputvariables1=list(inputdata)
del inputvariables1[7]
inputvariables1
inputdata1=inputdata[inputvariables1]

import statsmodels.formula.api as smf
import statsmodels.api as sm

inputdata1=sm.add_constant(inputdata1)
model1=sm.OLS(outputdata,inputdata1).fit()

model1.summary()











