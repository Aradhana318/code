# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
ToyotaCorolla =pd.read_csv(r"C:\Users\ANKIT\Desktop\Aradhana\Multilinear\ToyotaCorolla.csv",encoding='unicode_escape')

# to get top 6 rows
ToyotaCorolla.head() # to get top n rows use ToyotaCorolla.head(10)
ToyotaCorolla.columns

data=ToyotaCorolla[['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
data

# Correlation matrix
 
data.corr()

# we see there exists High collinearity between input variables especially between
 
# Scatter plot between the variables along with histograms
import seaborn as sns
plt.subplots(figsize=(10,6))
CorrelationMatrix=data.corr().abs()
sns.heatmap(CorrelationMatrix,annot=True)
plt.show()

#preparing model considering all the variables
#regression model
import statsmodels.formula.api as smf
#preparing model
model1=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data).fit()
model1.summary()

model2=smf.ols('Price~Doors',data=data).fit()
model2.summary()

model3=smf.ols('Price~Doors',data=data).fit()
model3.summary()

#preparing model withoutcc and doors only
model4=smf.ols('Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight',data=data).fit()
model4.summary()

import statsmodels.api as sm

sm.graphics.influence_plot(model1)

data1=data.drop(data.index[[80,960,221,660]],axis=0)
model5=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data1).fit()
model5.summary()
 
sm.graphics.influence_plot(model5)
data2=data1.drop(data1.index[[991,601,654]],axis=0)

model6=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data2).fit()
model6.summary()












