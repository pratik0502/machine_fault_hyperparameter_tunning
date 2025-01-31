import mlflow.data
import mlflow.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


import dagshub
dagshub.init(repo_owner='pratik0502', repo_name='machine_fault_hyperparameter_tunning', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/pratik0502/machine_fault_hyperparameter_tunning.mlflow')

data = pd.read_csv(r"D:\ML_ops\archive\data.csv")

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]

sd = StandardScaler()
x = sd.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
 


n_estimators = 40
max_depth = 10

mlflow.autolog()

mlflow.set_experiment('machine_fault_RD')

with mlflow.start_run(run_name='RD with artifcats'):

    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)



    # model = SVC(gamma=gm,kernel=kernel,C=c)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)

    CM = confusion_matrix(y_test,y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(CM, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix image
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    plt.close()

    CR = classification_report(y_test,y_pred)

    mlflow.log_metric('accuracy',acc)
    # mlflow.log_metric('Confusion_metrics',CM)
    # mlflow.log_metric('Confusion_report',CR)

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('max_depth',max_depth)

    mlflow.log_artifact(image_path)

    mlflow.sklearn.log_model(model,'SVC')
    mlflow.log_artifact(__file__)

    # train_data = x_train
    # train_data['fault'] = y_train

    # train_data = mlflow.data.from_padas(train_data)

    # mlflow.log_input(train_data,'training data')

    # test_data = x_test
    # test_data['fault'] = y_test

    # test_data = mlflow.data.from_pandas(test_data)

    # mlflow.log_input(test_data,'test_data')
    


    print('accuracy',acc)

    print('CM',CM) 

    print('CR',CR)


