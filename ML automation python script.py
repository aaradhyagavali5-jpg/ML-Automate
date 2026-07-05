import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler
import numpy as np
print('Welcome to Machine Learning Automated version for Regression and Classification')
dataset=input('Enter the dataset you want to use(path):')
orignal=pd.read_csv(f'{dataset}')
df=pd.DataFrame(orignal)
print(df.isnull().sum())
dcl=input('do you want to drop columns(yes/No):').lower()
if dcl=='yes':
    dropcols=input('Enter the columns you want to drop:').split(',')
    df.drop(dropcols,axis=1,inplace=True)
dropco=input('The null values in columns you want to drop(yes/No):').lower()
if dropco=='yes':
    dropcols=input('The null values in colums you want to drop:').split(',')
    df[dropcols].dropna(subset=dropcols,inplace=True)
else:
    print('No columns dropped')    

ropco=input(' you want to fill null values in  numeric columns (yes/No):').lower()


if ropco=='yes':
    cols=input("Enter the columns you want to fill with mean for numeric columns only (comma separated):").split(',')
    df[cols]=df[cols].fillna(df[cols].mean())    
else:
    print("Ok")    


rop=input(' you want to fill null values in categorical columns (yes/No):').lower()        
if rop=='yes':
    cols2=input("Enter the columns you want to fill with mode for categorical columns only (comma separated):").split(',')
    df[cols2]=df[cols2].fillna(df[cols2].mode()[0])
else:
    print("Ok")


while True:
            encoder=input('Enter the encoder you want to use (OrdianalEncoder,LabelEncoder,OneHotEncoder):').lower()
            if encoder=='labelencoder':
                column=input('Enter the column you want to labelencode:').split(',')
                le=LabelEncoder()
                for col in column:
                    df[col]=le.fit_transform(df[col])
                break
            elif encoder=='ordinalencoder':
                column=input('Enter the column you want to ordinalencode:').split(',')
                oe=OrdinalEncoder()
                df[column]=oe.fit_transform(df[column])        
                break
            elif encoder=='onehotencoder':
                column=input('Enter the column you want to onehotencode:').split(',')
                df=pd.get_dummies(df,columns=column) 
                break
            elif encoder=='all':
                column=input('Enter the column you want to labelencode:').split(',')
                if column[0]=="no":
                     continue
                else:
                     le=LabelEncoder()
                     for col in column:
                        df[col]=le.fit_transform(df[col])
                column=input('Enter the column you want to ordinalencode:').split(',')
                if column[0].strip()=="no":
                     continue
                else:
                    oe=OrdinalEncoder()
                    df[column]=oe.fit_transform(df[column])     
                column=input('Enter the column you want to onehotencode:').split(',')
                if column[0].strip()=="no":
                     continue
                else:
                    df=pd.get_dummies(df,columns=column)  
                break
            else:
                print('Invalid encoder')
    
              
target=input('Enter the target column:')        

x=df.drop(target,axis=1)
y=df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
std=StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.transform(x_test)

while True:
        model=input('Enter the model you want to use (LinearRegression,DecisionTrees):').lower()
        if model=='linearregression':
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
            model=LinearRegression()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
            print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
            print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")
            print(f"R^2 Score: {r2_score(y_test, y_pred)}")
            break
        elif model=='decisiontrees':
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
            model=DecisionTreeRegressor()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            print(f"classification report: {classification_report(y_test, y_pred)}")
            break



