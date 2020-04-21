# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
F_Startups =pd.read_csv(r"C:\Users\ANKIT\Desktop\Aradhana\Multilinear\F_Startups.csv")

# to get top 6 rows
F_Startups.head(10) # to get top n rows use F_Startups.head(10)
cleanup={"St":{"New York":"3","California":"1","Florida":"2"}}
F_Startups.replace(cleanup,inplace=True)
F_Startups.head()

# Correlation matrix 
F_Startups.corr()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(F_Startups)


# columns names
F_Startups.columns


F_Startups=F_Startups[['RD','Adm','Mkt','Pro']]
inputvariables=list(F_Startups)
del inputvariables[0]
inputvariables

outputvariables=list(F_Startups)[0]
outputvariables

inputdata=F_Startups[inputvariables]
inputdata


outputdata=F_Startups[[outputvariables]]

import statsmodels.formula.api as smf
import statsmodels.api as sm

inputdata=sm.add_constant(inputdata)
model=sm.OLS(outputdata,inputdata).fit()
model.summary()

sm.graphics.influence_plot(model)
inputdata1=inputdata.drop(inputdata.index[[15,14,36]],axis=0)
inputdata1
outputdata1=outputdata.drop(outputdata.index[[15,14,36]],axis=0)
outputdata1

import statsmodels.formula.api as smf
import statsmodels.api as sm

inputdata1=sm.add_constant(inputdata1)
model1=sm.OLS(outputdata1,inputdata1).fit()

model1.summary()





