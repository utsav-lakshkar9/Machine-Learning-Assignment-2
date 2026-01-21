import streamlit as st
import pandas as pd
import pickle
import st_yled
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,roc_auc_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

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
    if tp==0 and fp==0:
        return 0    
    return tp/(tp+fp)

def recall(y_true, y_pred,c):
    tp=true_positive(y_true,y_pred,c)
    fn=false_negative(y_true,y_pred,c)
    if tp==0 and fn==0:
        return 0    
    return tp/(tp+fn)

def f1(y_true, y_pred,c):
    p=precision(y_true,y_pred,c)
    r=recall(y_true,y_pred,c)    
    if p==0 and r==0:
        return 0    
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

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(page_title="Multiclass Cancer Risk Classification from Simulated Patient Data",layout="wide")

st.title("Multiclass Cancer Risk Classification from Simulated Patient Data",text_alignment="center")
st.markdown("<div style='text-align:center'>"
    "A multiclass classification system to predict Cancer Type (Lung, Breast, Colon, Prostate, Skin) from simulated demographic, lifestyle, environmental, and genetic risk factors"
    "</div>",unsafe_allow_html=True)

# Initialize st_yled
st_yled.init('style.css')

# -------------------------------
# Load models and feature names
# -------------------------------
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features(feature_path):
    with open(feature_path, "rb") as f:
        return pickle.load(f)

# Update paths as per your repo
MODEL_PATHS = {
    "Logistic Regression": "LogisticRegression.pkl",
    "Decision Tree": "DecisionTreeClassifier.pkl",
    "KNN Classifier": "KNeighborsClassifier.pkl",
    "Random Forest": "RandomForestClassifier.pkl",
    "Naive Bayes" : "GaussianNB.pkl",
    "XGBoost": "XGBClassifier.pkl",
}

FEATURES_PATH = "feature_names.pkl"

feature_names = load_features(FEATURES_PATH)
models=list(MODEL_PATHS.keys())
models.append("Combined View")

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox("Select a trained model",models)

# -------------------------------
# Dataset upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)
with open("./cancer-risk-factors.csv", "rb") as f:
    csv_data = f.read()

st.download_button("Download the training data",csv_data,"Training-data.csv",mime="text/csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Feature validation
    # -------------------------------
    missing_features = set(feature_names) - set(df.columns)
    extra_features = set(df.columns) - set(feature_names)

    if missing_features:
        st.error(f"❌ Missing required features: {missing_features}")
        st.stop()

    if extra_features:
        st.warning(f"⚠️ Extra columns ignored: {extra_features}")

    # Align feature order
    X = df[feature_names]

    st.header("Test Dataset Preview",text_alignment="center")
    st.table(X.head())

    # Target (if available)
    y_true = None
    if "Cancer_Type" in df.columns:
        y_true = df["Cancer_Type"]
    

    X_train,X_test,y_train,y_test=train_test_split(X,y_true,test_size=0.6,stratify=y_true,random_state=101)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    lst=list(y_train.unique())


    # -------------------------------
    # Prediction
    # -------------------------------
    predictions=[]
    logreg = load_model(MODEL_PATHS["Logistic Regression"]) 
    y_pred_log=logreg.predict(X_test)
    predictions.append(y_pred_log)
    
    dtc=load_model(MODEL_PATHS["Decision Tree"])
    dtc.fit(X_train,y_train)
    y_pred_dtc=dtc.predict(X_test)
    predictions.append(y_pred_dtc)

    knn = load_model(MODEL_PATHS["KNN Classifier"])
    knn.fit(X_train,y_train)
    y_pred_knn=knn.predict(X_test)
    predictions.append(y_pred_knn)

    gnb = load_model(MODEL_PATHS["Random Forest"])
    gnb.fit(X_train,y_train)
    y_pred_gnb=gnb.predict(X_test)
    predictions.append(y_pred_gnb)

    rfc=load_model(MODEL_PATHS["Naive Bayes"])
    rfc.fit(X_train,y_train)
    y_pred_rfc=rfc.predict(X_test)
    predictions.append(y_pred_rfc)

    xgb=load_model(MODEL_PATHS["XGBoost"])
    le=LabelEncoder()
    le.fit_transform(y_train)
    y_pred_xgb=le.inverse_transform(xgb.predict(X_test))
    predictions.append(y_pred_xgb)

    if model_name!="Combined View":
        model = load_model(MODEL_PATHS[model_name])
        if model_name!='XGBoost':
            y_pred = model.predict(X_test)
        else:
            le=LabelEncoder()
            le.fit_transform(y_true)
            y_pred = le.inverse_transform(model.predict(X_test))
            #print(y_pred)

        st.header("Prediction Output",text_alignment="center")
        pred_counts = (pd.Series(y_pred).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="Count"))
        st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

    else:
        
        st.header("Prediction Output",text_alignment="center")
        
        col1,col2,col3,col4,col5,col6 = st.columns(6)

        pred_counts_log = (pd.Series(y_pred_log).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="Logistic Regression"))
        with col1:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_log.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

        pred_counts_dtc = (pd.Series(y_pred_dtc).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="Decision Tree Classifier"))
        with col2:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_dtc.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

        pred_counts_knn = (pd.Series(y_pred_knn).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="KNN Classifier"))
        with col3:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_knn.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

        pred_counts_gnb = (pd.Series(y_pred_gnb).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="Naive Bayes Classifier"))
        with col4:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_gnb.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

        pred_counts_rfc = (pd.Series(y_pred_rfc).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="Random Forest Classifier"))
        with col5:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_rfc.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

        pred_counts_xgb = (pd.Series(y_pred_xgb).value_counts().rename_axis("Predicted Cancer Type").reset_index(name="XGBoost Classifier"))
        with col6:
            st.markdown(f"""<div style="display: flex; justify-content: center;">{pred_counts_xgb.to_html(index=False,justify="center")}</div>""",
        unsafe_allow_html=True)

    # -------------------------------
    # Evaluation metrics
    # -------------------------------
    if y_true is not None:
        st.header("Model Evaluation Metrics",text_alignment="center")

        if model_name!="Combined View":    

            col1, col2, col3, col4, col5, col6= st.columns(6)

            col1.metric("Accuracy",f"{macro_accuracy(y_test, y_pred,lst):.3f}")

            col2.metric("Precision",f"{macro_precision(y_test, y_pred,lst):.3f}")

            col3.metric("Recall",f"{macro_recall(y_test, y_pred,lst):.3f}")

            col4.metric("F1",f"{macro_f1(y_test, y_pred,lst):.3f}")

            col5.metric("MCC",f"{matthews_corrcoef(y_test,y_pred):.3f}")

            col6.metric("AUC",f"{roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr'):.3f}")

        else:

            for i in range(len(predictions)):
                st.subheader(models[i],text_alignment="center")
                col1, col2, col3, col4, col5, col6= st.columns(6)
                y_pred=predictions[i]
                model=load_model(MODEL_PATHS[models[i]])
                col1.metric("Accuracy",f"{macro_accuracy(y_test, y_pred,lst):.3f}")

                col2.metric("Precision",f"{macro_precision(y_test, y_pred,lst):.3f}")

                col3.metric("Recall",f"{macro_recall(y_test, y_pred,lst):.3f}")

                col4.metric("F1",f"{macro_f1(y_test, y_pred,lst):.3f}")

                col5.metric("MCC",f"{matthews_corrcoef(y_test,y_pred):.3f}")

                col6.metric("AUC",f"{roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr'):.3f}")
        
        # -------------------------------
        # Classification report
        # -------------------------------
        st.header("Classification Report",text_alignment="center")
        if model_name!="Combined View":
            report = classification_report(y_test, y_pred, output_dict=True)
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                report_df = pd.DataFrame(report).transpose().reset_index()
                report_df.rename(columns={"index": "Class"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)

        else:
            col1,col2,col3,col4,col5= st.columns([2,1,2,1,2])                        

            report_log = classification_report(y_test, y_pred_log, output_dict=True)
            with col1:
                report_df_log = pd.DataFrame(report_log).transpose().reset_index()
                report_df_log.rename(columns={"index": "Logistic Regression"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_log.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)

            report_dtc = classification_report(y_test, y_pred_dtc, output_dict=True)
            with col3:
                report_df_dtc = pd.DataFrame(report_dtc).transpose().reset_index()
                report_df_dtc.rename(columns={"index": "Decision Tree"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_dtc.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)
            
            report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
            with col5:
                report_df_knn = pd.DataFrame(report_knn).transpose().reset_index()
                report_df_knn.rename(columns={"index": "KNN Classifier"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_knn.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)
            
            report_gnb = classification_report(y_test, y_pred_gnb, output_dict=True)
            with col1:
                report_df_gnb = pd.DataFrame(report_gnb).transpose().reset_index()
                report_df_gnb.rename(columns={"index": "Naive Bayes Classifier"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_gnb.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)
            
            report_rfc = classification_report(y_test, y_pred_rfc, output_dict=True)
            with col3:
                report_df_rfc = pd.DataFrame(report_rfc).transpose().reset_index()
                report_df_rfc.rename(columns={"index": "Random Forest Classifier"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_rfc.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)
            
            report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
            with col5:
                report_df_xgb = pd.DataFrame(report_xgb).transpose().reset_index()
                report_df_xgb.rename(columns={"index": "Logistic Regression"}, inplace=True)
                st.markdown(
                f"""<div style="display: flex; justify-content: center;">{report_df_xgb.to_html(index=False,justify="center")}</div>""",
                unsafe_allow_html=True)                        
            
        # -------------------------------
        # Confusion matrix
        # -------------------------------
        st.header("Confusion Matrix",text_alignment="center")
        if model_name!="Combined View":
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,4),dpi=100)
            sns.heatmap(cm,annot=True,fmt='g',annot_kws={"size": 9},cbar_kws={"shrink": 0.8},xticklabels=sorted(y_test.unique()),yticklabels=sorted(y_test.unique()))
            ax.set_title(model_name, fontsize=13, pad=20)
            ax.set_xlabel("Predicted",fontsize=10)
            ax.set_ylabel("Actual",fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize=9)
            fig.tight_layout()
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig)            
        
        else:
            col1,col2,col3= st.columns(3)
            figures=[]
            for i in range(len(predictions)):
                y_pred=predictions[i]                
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5,4),dpi=100)
                sns.heatmap(cm,annot=True,fmt='g',annot_kws={"size": 9},cbar_kws={"shrink": 0.8},xticklabels=sorted(y_test.unique()),yticklabels=sorted(y_test.unique()))
                ax.set_title(models[i], fontsize=13, pad=20)
                ax.set_xlabel("Predicted",fontsize=10)
                ax.set_ylabel("Actual",fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize=9)
                fig.tight_layout()
                figures.append(fig)
            
            with col1:
                st.pyplot(figures[0])
            with col2:
                st.pyplot(figures[1])
            with col3:
                st.pyplot(figures[2])
            with col1:
                st.pyplot(figures[3])
            with col2:
                st.pyplot(figures[4])
            with col3:
                st.pyplot(figures[5])
            

    else:
        st.info(
            "Ground truth label `Cancer_Type` not found. "
            "Evaluation metrics are skipped."
        )
