import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline 

heart_data= pd.read_csv('/content/heart_failure_clinical_records.csv')
heart_data 

heart_data.head(20) 


heart_data.info() 


heart_data.shape 

heart_data.isnull().sum() 

heart_data.nunique()
heart_data.describe() 

plt.figure()
sns.heatmap(heart_data.corr(),annot = True)
plt.title('correlation heatmap')
plt.show()
 
countFemale = len((heart_data[heart_data.sex == 0]))
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100))) 

countmale = len((heart_data[heart_data.sex == 1]))
print("Percentage of male Patients: {:.2f}%".format((countmale / (len(df.sex))*100))) 


age_column = 'age'
diabetes_column = 'diabetes'

sns.set_style('whitegrid')


sns.distplot(heart_data[age_column], kde=False,  color='blue',bins=20)
plt.xlabel("Age")
plt.ylabel("diabetes")
plt.title("Age Distribution in Heart Data")

plt.show() 


# how does smoking affect on high blood pressure
sns.violinplot(data=heart_data, x='smoking', y='high_blood_pressure', palette='viridis')
plt.title('Violinplot')
plt.show() 


# how many people have age related anamenia


fig = px.histogram(heart_data, x="age", y="anaemia", color="age", barmode="group", height=400)
fig.show() 


fig = px.scatter_3d(heart_data, x='creatinine_phosphokinase', y='serum_creatinine', z='serum_sodium',
              color='anaemia' ,hover_name="age")
fig.show() 


totaldeath = px.histogram(heart_data, x="age", y="DEATH_EVENT",
             color='sex', barmode='group',
             height=400)
totaldeath.show() 

smokers_death = heart_data[heart_data['smoking'] == 1]
plt.figure(figsize=(10, 6))
plt.hist(smokers_death['age'], bins=20, edgecolor='blue')
plt.title('Distribution of Age for Smokers')
plt.xlabel('Age')
plt.ylabel('Death_count')
plt.show()


#platelet count distribution in age group
fig = px.histogram(heart_data, x="age", y="platelets", color="sex")
fig.show()


x = heart_data[['age','anaemia','diabetes','sex','smoking']]
y = heart_data['DEATH_EVENT'] 

x 
Y 

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size = 0.2, random_state = 0) 

from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators = 100 ) 

rf.fit(xtrain, ytrain) 

predictions = rf.predict(xtest)
predictions 

from sklearn.metrics import accuracy_score
accuracy_score(predictions, ytest) 


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators" : [100,200],
  "max_depth" : [None,5,10],
  "min_samples_split" : [2,5],
  "min_samples_leaf" : [1,2,4],
  "criterion" : ["gini", "entropy" ]
}

classifier = RandomForestClassifier()
grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid)
grid_search.fit(xtrain, ytrain) 


print(grid_search.best_params_) 

y_pred1 = classifier3.predict(xtest)
y_pred1 

accuracy_score(y_pred1, ytest)
