#!/usr/bin/env python
# coding: utf-8

# # Machine Learning in Business: OilyGiant Mining Location Discovery. 

# Project will demonstrate ability to perform machine learning in a business setting.  Data comes from OilyGiant mining company (geo_data_0/1/2.csv which has three feature points). Target task is to find the best place for a new well.
# 
# Conditions to Assist in Model. :
# Only linear regression is suitable for model training.
# A study of 500 points is carried with picking the best 200 points for the profit calculation.
# The budget for development of 200 oil wells is 100 USD million.
# One barrel of raw materials brings 4.5 USD of revenue The revenue from one unit of product is 4,500 dollars (volume of reserves is in thousand barrels).
# After the risk evaluation, keep only the regions with the risk of losses lower than 2.5%. From the ones that fit the criteria, the region with the highest average profit should be selected.
# 
# Procedure:
# Collect the oil well parameters in the selected region: oil quality and volume of reserves;
# Build a model for predicting the volume of reserves in the new wells.  Analyze to pick region with highest profit margin.
# Pick the oil wells with the highest estimated values;
# Pick the region with the highest total profit for the selected oil wells.
# Data is on oil samples from three regions. Parameters of each oil well in the region are already known. Analyze potential profit and risks using the Bootstrapping technique.
# 

# In[1]:


#import all needed libraries 
import pandas as pd
import numpy as np

#import display libraries 
import seaborn as sns
import matplotlib.pyplot as plt

#import named regression models 
from sklearn.linear_model import LinearRegression

#import ability to split into training and testing data sets 
from sklearn.model_selection import train_test_split

#import ability to evaluate accuracy of data 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


from joblib import dump

#needed to compare. 
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



# EDA of files

# In[2]:


site1=pd.read_csv('/datasets/geo_data_0.csv')
site1.head(3)


# In[3]:


site1.info()


# In[4]:


#check for empty column values
site1.isna().sum()


# In[5]:


#check for duplicates 
print(site1['id'].duplicated().sum())


# In[6]:


#confirm correct column names 
site1.columns


# In[7]:


# Drop duplicates in the DataFrame itself
site1 = site1.drop_duplicates(subset=['id'])

# Confirm duplicates were dropped
print(site1['id'].duplicated().sum())  


# In[8]:


print(site1.shape)
print(site1.columns)


# In[9]:


site2=pd.read_csv('/datasets/geo_data_1.csv')
site2.head(3)


# In[10]:


site2.info()


# In[11]:


print(site2.head(3))


# In[12]:


site2.isna().sum()


# In[13]:


site2.duplicated().sum()


# In[14]:


site2.shape


# In[15]:


site3=pd.read_csv('/datasets/geo_data_2.csv')
site3.head(3)


# In[16]:


site3.info()


# In[17]:


site3.isna().sum()


# In[18]:


site3.duplicated().sum()


# In[19]:


site3.shape


# 3. Profit calculation prep

# In[20]:


sites=[site1,site2,site3]


# for i in range(3):
#     sites[i] = sites[i].drop(['id'],axis=1)
#     display(sites[i])
# 

# Confirmed that cleared each dataframe of "id" column. 

# In[21]:


#define variables from source data 
revenue_per_thousand_barrels = 100000000 
revenue_per_well = 4500 
wells=200

# Calculate break-even volume
break_even_volume = revenue_per_thousand_barrels/revenue_per_well

#calculate average break-even volume for each well
average_vol_each_well= break_even_volume/wells

print(f'Break-even volume needed: {break_even_volume}')
print(f'Average volume required per well to break-even: {average_vol_each_well}')



# 4. Write functions to calculate profit, bootstrap, and rmse from a set of selected oil wells and model predictions:

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# Success. This snippet accurately calculates the break-even volume and the average volume required per well to break even. The use of constants like revenue_per_thousand_barrels and revenue_per_well ensures clarity and maintains scalability for future changes.
# </div>
# 

# In[22]:


def profit_calculation(target, predictions, count):
    # Sort predictions in descending order
    probs_sorted = predictions.sort_values(ascending=False)    
    # Select top 'count' values using valid indices
    selected = target[probs_sorted.index][:count]
    return revenue_per_well * selected.sum() - revenue_per_thousand_barrels


# In[23]:


#rmse calculation 
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))


# In[24]:


#from 
def bootstrap_profit(valid, predictions, wells, profit_calculation, n_samples=1000, random_state=None):
    values = []
    assert (valid.index == predictions.index).all()
    for _ in range(n_samples):
        target_ranked = valid['product'].sample(n=500, replace=True, random_state=12345)
        probability_ranked = predictions.iloc[target_ranked.index]      
        values.append(profit_calculation(target_ranked, probability_ranked, wells))

    values = pd.Series(values)
    return {
        'mean': values.mean(),
        'low_quantile': values.quantile(0.025),
        'high_quantile': values.quantile(0.975),
        'risk_of_loss': (values < 0).mean() * 100
    }


# In[25]:


#profit computation 
def calculate_profit(valid, predictions, wells, profit_calculation):
    return profit_calculation(valid['product'], predictions, wells)


# 5. Calculate risk and profit for each region 

# In[26]:


for i, site in enumerate(sites):
    #drop unnecessary column 'id'
    site = site.drop(columns=['id'])
    display(site)
    
    train, valid = train_test_split(site, test_size=0.25, random_state=12345)
    train = train.reset_index()
    valid = valid.reset_index()

    # Train model
    model = LinearRegression()
    model.fit(train.drop(['product'], axis=1), train['product'])
    predictions = pd.Series(model.predict(valid.drop(['product'], axis=1)), index=valid.index)
    
    # assert valid.index == predictions.index

    print(f'Information for Site {i}')

    # Calculate and display RMSE
    rmse = calculate_rmse(valid['product'], predictions)
    print(f'RMSE:', rmse)

    # Average predicted reserves
    average = predictions.mean()
    print(f'Predicted Reserves Average Volume:', average)

    # Calculate and display profit
    profit = calculate_profit(valid, predictions, wells, profit_calculation)
    print(f'Profit: {profit}')

    # Bootstrap analysis
    stats = bootstrap_profit(valid, predictions, wells, profit_calculation, n_samples=1000, random_state=12345)
    print(f'Average profit:', stats['mean'])
    print(f'2.5% quantile:', stats['low_quantile'])
    print(f'97.5% quantile:', stats['high_quantile'])
    print(f'Risk of loss:', stats['risk_of_loss'], '%')
    print('******************************************************************')
    print()


# Project Summary 
# 
# Cleaning of each dataframe had to be performed independently.  Once completed tasks could be grouped together.  Each dataframe needed the "id" column dropped before running through the model.  All other tasks were able to be placed in a loop where each dataframe was looped through. 
# 
# In order to break-even 22,222.22 barrels of oil need to be generated at a site.  Average volume required per well to break-even is 111.11 barrells. 
# 
# Site 1 has the lowest risk of loss among all the evaluated sites.  Site 1 has the largest average profit at 5 million dollars.  Site 1 is also the only site where the lower quantile is a positive number.  It is notebwrothy that Site 1 has the lowest predicted average volume of reserves by almost 30 barrels.  However Site 1 remains the best business decision because its extreme low risk of failure.  The expected 24 million dollars generation from this site could fund exploration of sites besides site 0 and site 2 for Oily Giant.   
