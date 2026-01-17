import pickle
import streamlit as st
from flask import Flask,render_template
from pathlib import Path


app=Flask(__name__)
models = []
# Define the directory path and extension
directory_path = Path('.') # Current directory. Replace '.' with your specific path
extension = '*.pkl'       # The extension you are looking for

# Iterate over all files matching the pattern in the current directory (non-recursive)
for file_path in directory_path.glob(extension):
    with open(file_path, 'rb') as file:
        #print(file.name)
        model=pickle.load(file)
        #print(type(model))
        models.append(model)

#print("Models",models)

@app.route("/")
def home():
    option = st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option)
    return render_template('index.html')


if __name__=="__main__":        
    app.run(debug=True)