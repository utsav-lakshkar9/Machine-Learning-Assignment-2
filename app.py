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

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(
    page_title="Multiclass Cancer Risk Classification from Simulated Patient Data",
    layout="wide")

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

    st.subheader("Test Dataset Preview",text_alignment="center")
    st.dataframe(X.head())

    # Target (if available)
    y_true = None
    if "Cancer_Type" in df.columns:
        y_true = df["Cancer_Type"]
    

    X_train,X_test,y_train,y_test=train_test_split(X,y_true,test_size=0.3,stratify=y_true,random_state=101)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    lst=list(y_train.unique())

    # -------------------------------
    # Prediction
    # -------------------------------
    if model_name!="Combined View":
        model = load_model(MODEL_PATHS[model_name])
        if model_name!='XGBoost':
            y_pred = model.predict(X_test)
        else:
            le=LabelEncoder()
            le.fit_transform(y_true)
            y_pred = le.inverse_transform(model.predict(X_test))
            #print(y_pred)

        st.subheader("Prediction Output",text_alignment="center")
        pred_counts = pd.Series(y_pred, name="Predicted Cancer Type").value_counts().to_frame("Count").reset_index()
        st.markdown(
            f"""<div style="display: flex; justify-content: center;">{pred_counts.to_html(index=False,justify="center")}</div>""",
            unsafe_allow_html=True)

        #pred_counts = pd.Series(y_pred, name="Predicted Cancer Type").value_counts().to_frame("Count")
        #col1, col2, col3 = st.columns([1, 2, 1])
        #with col2:
        #    st.table(pred_counts)

        #col1, col2, col3 = st.columns([1, 1, 1])
        #with col2:
        #    st.markdown("<table style='text-align:center'>",pd.Series(y_pred, name="Predicted Cancer Type").value_counts(),"</div>",unsafe_allow_html=True)

    # -------------------------------
    # Evaluation metrics
    # -------------------------------
    if y_true is not None:
        st.subheader("Model Evaluation Metrics",text_alignment="center")

        col1, col2, col3, col4, col5, col6= st.columns(6)

        col1.metric("Accuracy",f"{macro_accuracy(y_test, y_pred,lst)*100:.3f}")

        col2.metric("Precision",f"{macro_precision(y_test, y_pred,lst)*100:.3f}")

        col3.metric("Recall",f"{macro_recall(y_test, y_pred,lst)*100:.3f}")

        col4.metric("F1",f"{macro_f1(y_test, y_pred,lst)*100:.3f}")

        col5.metric("MCC",f"{matthews_corrcoef(y_test,y_pred)*100:.3f}")

        col6.metric("AUC",f"{roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr')*100:.3f}")


        # -------------------------------
        # Classification report
        # -------------------------------
        st.subheader("Classification Report",text_alignment="center")
        report = classification_report(y_test, y_pred, output_dict=True)
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            report_df = pd.DataFrame(report).transpose().reset_index()
            report_df.rename(columns={"index": "Class"}, inplace=True)
            st.markdown(
            f"""<div style="display: flex; justify-content: center;">{report_df.to_html(index=False,justify="center")}</div>""",
            unsafe_allow_html=True)


        # -------------------------------
        # Confusion matrix
        # -------------------------------
        st.subheader("Confusion Matrix",text_alignment="center")
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
        #st.pyplot(fig,use_container_width=False)

    else:
        st.info(
            "Ground truth label `Cancer_Type` not found. "
            "Evaluation metrics are skipped."
        )
