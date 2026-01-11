import pickle
from flask import Flask,render_template

app=Flask(__name__)
with open('model.pkl','rb') as file:
    model = pickle.load(file)
@app.route("/")
def home():
    return render_template('index.html')

if __name__=="__main__":        
    app.run(debug=True)