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

# Building the model
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy')

# Training the model
extra_tree_forest.fit(X, y)

# Computing the importance of each feature
feature_importance = extra_tree_forest.feature_importances_

# Normalizing the individual importances
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
models.append(dtc)
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

def true_positive(y_true, y_pred,c):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == c and yp == c:
            tp += 1
    return tp

def true_negative(y_true, y_pred,c):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt !=c and yp != c:
            tn += 1
    return tn

def false_positive(y_true, y_pred,c):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt != c and yp == c:
            fp += 1
    return fp

def false_negative(y_true, y_pred,c):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == c and yp != c:
            fn += 1
    return fn

def accuracy(y_true, y_pred,c):
    tp=true_positive(y_true,y_pred,c)
    tn=true_negative(y_true,y_pred,c)
    fp=false_positive(y_true,y_pred,c)
    fn=false_negative(y_true,y_pred,c)
    return (tp+tn)/(tp+tn+fp+fn)

def precision(y_true, y_pred,c):
    tp=true_positive(y_true,y_pred,c)
    fp=false_positive(y_true,y_pred,c)    
    return tp/(tp+fp)

def recall(y_true, y_pred,c):
    tp=true_positive(y_true,y_pred,c)
    fn=false_negative(y_true,y_pred,c)    
    return tp/(tp+fn)

def f1(y_true, y_pred,c):
    p=precision(y_true,y_pred,c)
    r=recall(y_true,y_pred,c)    
    return 2*p*r/(p+r)

def macro_accuracy(y_true, y_pred, lst):
    val=0
    for i in lst:
        val+=accuracy(y_true,y_pred,i)
    return val/len(lst)

def macro_precision(y_true, y_pred, lst):
    val=0
    for i in lst:
        val+=precision(y_true,y_pred,i)
    return val/len(lst)

def macro_recall(y_true, y_pred, lst):
    val=0
    for i in lst:
        val+=recall(y_true,y_pred,i)
    return val/len(lst)

def macro_f1(y_true, y_pred, lst):
    val=0
    for i in lst:
        val+=f1(y_true,y_pred,i)
    return val/len(lst)

lst=list(y_train.unique())
feature_names = X.columns.tolist()
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

for i in range(len(model_names)):
    cm = confusion_matrix(np.array(y_test),predictions[i])
    sns.heatmap(cm,annot=True,fmt='g',xticklabels=sorted(y_test.unique()),yticklabels=sorted(y_test.unique()))
    plt.ylabel('Actual', fontsize=13)
    plt.title(model_names[i], fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()
    plt.show()
    model=models[i]
    y_pred_m=predictions[i]

    print("\nFor ",model_names[i])
    print("Accuracy: ",macro_accuracy(y_test,y_pred_m,lst)*100)
    print("Precision: ",macro_precision(y_test,y_pred_m,lst)*100)
    print("Recall: ",macro_recall(y_test,y_pred_m,lst)*100)
    print("F1: ",macro_f1(y_test,y_pred_m,lst)*100)
    print("MCC: ",matthews_corrcoef(y_test,y_pred_m)*100)
    print("AUC: ",roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr')*100)

    with open(model_names[i]+'.pkl','wb') as file:
        pickle.dump(model,file)
    