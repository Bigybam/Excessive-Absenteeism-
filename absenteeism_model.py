#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[194]:


### importing libraries

import numpy as np
import pandas as pd
import pickle 

class absenteeism_model():
    def __init__ (self, model, scaler):
      with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
            
    def load_and_clean_data(self,datafile):
        df = pd.read_csv(datafile, delimiter=',')
        df = df.drop(['ID'], axis=1)
        
        # working on date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
       
    # Relevant functions for month and day of the week
        def to_month(n):
            return n.month
        def to_weekday(n):
            return n.weekday()
        
        df['Month value'] = df['Date'].apply(to_month)
        df["Day of the week"] = df['Date'].apply(to_weekday)
        
        df = df.drop(['Date'], axis=1)
        
        # Working on Education{0: lower education, 1: Higher education}
       
        df['Education'] = df['Education'].map({1:0,2:1,3:1,4:1}) 
        
        
        # working on Absenteeism
        df['Absenteeism Time in Hours'] = 'NaN'
        
        df = df.drop(['Absenteeism Time in Hours'], axis = 1)
        
        
        #working on Reasons
         
        reasons = pd.get_dummies(df['Reason for Absence'])
        
        reason_type1 = reasons.loc[:,1:14].max(axis = 1)
        reason_type2 = reasons.loc[:,15:17].max(axis = 1)
        reason_type3 = reasons.loc[:,18:21].max(axis = 1)
        reason_type4 = reasons.loc[:,22:28].max(axis = 1)
        
        
        df = pd.concat([df, reason_type1,reason_type2,reason_type3,reason_type4], axis = 1)
        
        
        df= df.drop(['Reason for Absence'], axis = 1)
        
        column_rename = [  'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Month value', 'Day of the week',
        'reason_type1', 'reason_type2', 'reason_type3', 'reason_type4']
        
        
        df.columns = column_rename
        
        column_reordered = ['reason_type1', 'reason_type2', 'reason_type3', 'Day of the week',
        'reason_type4', 'Month value','Transportation Expense', 'Distance to Work', 'Age',
        'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets']
        
        df = df[column_reordered]
        
        
            # drop the variables we decide we don't need
        df = df.drop(['Day of the week','Daily Work Load Average','Distance to Work'],axis=1)
        
        df = df.fillna(value = 0)
        
        self.preprocessed_data = df.copy()
        
        self.data = self.scaler.transform(df)
        
        
        
        
        
          # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
    def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data


# In[195]:


def predict (model, scaler, dataset):
    model = absenteeism_model(model = model, scaler = scaler)
    model.load_and_clean_data(datafile= dataset)
    result = model.predicted_outputs()
    return result


# In[ ]:




