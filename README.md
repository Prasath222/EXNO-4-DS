# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1408" height="416" alt="image" src="https://github.com/user-attachments/assets/c82bebf6-454f-45df-aebe-089dd56c2ecf" />

```
data.isnull().sum()
```

<img width="225" height="487" alt="image" src="https://github.com/user-attachments/assets/addbcebc-720a-455b-9627-7de3d75c4a2d" />

```
missing=data[data.isnull().any(axis=1)] 
missing
```

<img width="1360" height="420" alt="image" src="https://github.com/user-attachments/assets/faa6525e-5083-4f7c-9028-d9c093a01f76" />

```
data2=data.dropna(axis=0) 
data2
```

<img width="1392" height="414" alt="image" src="https://github.com/user-attachments/assets/2ef0bce5-13d5-452f-bb9f-6dd7522792fc" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1139" height="324" alt="image" src="https://github.com/user-attachments/assets/40158dec-bf94-43a9-8d71-9ee72805e8ef" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="365" height="418" alt="image" src="https://github.com/user-attachments/assets/0fff5f4c-4d71-42e5-b98d-29cc4fce4fc2" />

```
data2
```

<img width="1280" height="419" alt="image" src="https://github.com/user-attachments/assets/6d6e8b7d-ccab-45c0-a380-809875a26e38" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1639" height="274" alt="image" src="https://github.com/user-attachments/assets/b22b9f35-7fe1-4130-9299-1a2a0290958d" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1448" height="36" alt="image" src="https://github.com/user-attachments/assets/fa39255c-b5c0-4ead-af26-e48b08192dcc" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1441" height="32" alt="image" src="https://github.com/user-attachments/assets/cc6e1112-9b53-4af0-864f-e138ebf44422" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="191" height="32" alt="image" src="https://github.com/user-attachments/assets/8eb6e04f-0252-4b54-b742-3c89bfc8bd90" />

```
x=new_data[features].values
print(x)
```

<img width="399" height="135" alt="image" src="https://github.com/user-attachments/assets/35c9144d-8017-453f-8d3d-49c05b94baf5" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="292" height="75" alt="image" src="https://github.com/user-attachments/assets/e5e8dd35-fa12-4a68-8839-705685c64e4c" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="177" height="47" alt="image" src="https://github.com/user-attachments/assets/08aa40a8-cdeb-445c-8975-f976f664e6f1" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="189" height="27" alt="image" src="https://github.com/user-attachments/assets/83544dc2-7f1a-4c3c-9a94-1863d22976c9" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="270" height="29" alt="image" src="https://github.com/user-attachments/assets/8f924df6-e4f3-4e3f-acb1-9cc8f07a65d7" />

```
data.shape
```

<img width="136" height="30" alt="image" src="https://github.com/user-attachments/assets/ad90ff59-d647-4ff2-9f76-e584a8b58ad9" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1518" height="75" alt="image" src="https://github.com/user-attachments/assets/2d1222dc-06e1-488a-b7dc-2b5e2120cd4a" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="391" height="169" alt="image" src="https://github.com/user-attachments/assets/fb38c585-c936-443c-9e3a-c3b92bc81405" />

```
tips.time.unique()
```

<img width="325" height="38" alt="image" src="https://github.com/user-attachments/assets/6fdae544-228c-4fff-91b0-1687fe51f195" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="202" height="73" alt="image" src="https://github.com/user-attachments/assets/9f3ec504-b665-449f-b918-83730629375f" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="333" height="36" alt="image" src="https://github.com/user-attachments/assets/933a9e19-3cac-4d0a-b500-9fcfcfc47d80" />

# RESULT:

Thus, Feature selection and Feature scaling has been used on the given dataset.

