import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef,roc_auc_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

data=pd.read_csv(r'./cancer-risk-factors.csv')
risk_weight={'Low':0,'Medium':1,'High':2}
data['Risk_Level']=data['Risk_Level'].map(risk_weight)
X=data.drop(columns=['Patient_ID','Cancer_Type'])
y=data['Cancer_Type']

extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy')
extra_tree_forest.fit(X, y)
feature_importance = extra_tree_forest.feature_importances_
feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_],axis = 0)
feature_mapping={}
for i in range(len(X.columns)):
    if feature_importance_normalized[i]>=0.005:
        feature_mapping[X.columns[i]]=feature_importance_normalized[i]

dropped_columns=[i for i in X.columns if i not in feature_mapping]
X.drop(columns=dropped_columns,inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=101)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
models=[]
predictions=[]

logreg = LogisticRegression(solver="lbfgs", C=5) 
logreg.fit(X_train,y_train)
y_pred_log=logreg.predict(X_test)
models.append(logreg)
predictions.append(y_pred_log)

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc=dtc.predict(X_test)
models.append(dtc)
predictions.append(y_pred_dtc)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)
models.append(knn)
predictions.append(y_pred_knn)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb=gnb.predict(X_test)
models.append(gnb)
predictions.append(y_pred_gnb)

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred_rfc=rfc.predict(X_test)
models.append(rfc)
predictions.append(y_pred_rfc)

xgb=XGBClassifier(objective="multi:softprob",num_class=5,eval_metric="mlogloss")
le=LabelEncoder()
xgb.fit(X_train,le.fit_transform(y_train))
y_pred_xgb=le.inverse_transform(xgb.predict(X_test))
models.append(xgb)
predictions.append(y_pred_xgb)

model_names=[i.__class__.__name__ for i in models]
feature_names = X.columns.tolist()
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

for i in range(len(model_names)):
    model=models[i]
    y_pred_m=predictions[i]
    with open(model_names[i]+'.pkl','wb') as file:
        pickle.dump(model,file)
    