#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os
import tensorflow 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///emo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Emo(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(500), nullable=False)
    file_path = db.Column(db.String(1000), nullable=False)
    pred = db.Column(db.String(500), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.name}"






model =load_model("model_weights.h5")
print("--------------MODEL LOAD SUCCESSFULLY--------------------")


labels =  ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


labels_names = []
for i in range(len(labels)):
  labels_names += [i]
reverse_mapping = dict(zip(labels_names, labels)) 
def mapper(value):
  return reverse_mapping[value]

def pred_emo(imagefromdb):
    # Load the image in grayscale
    test_image = cv2.imread(imagefromdb, cv2.IMREAD_GRAYSCALE)
    # Resize the image to (48, 48)
    test_image = cv2.resize(test_image, (48, 48))
    # Expand the dimensions of the image to match the model's input shape
    test_image = np.expand_dims(test_image, axis=-1)
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    # Normalize the image
    test_image = test_image / 255.0
    # Load the model
    model = load_model("model.h5")
    # Make a prediction
    mresult = model.predict(test_image).round(3)
    aresult = np.argmax(mresult)
    result = mapper(aresult)
    print(f'Prediction is {result}.')
    return result



@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    

  
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] 
        name = request.form.get('name')
        filename = file.filename   
        print("USERNAME: ",name)
        print("Input Image File Name = ", filename)
        file_path = os.path.join('static/user uploaded/', filename)
        file.save(file_path)
        print("Input Image File Path = ", file_path)




        print("--------USER UPLOADED SUCCESSFULLY ---------- ")  


        print("------PREDICTIONS START--------")
        pred = pred_emo(imagefromdb=file_path)

        print("prediction result is :", pred)


        emo = Emo(name=name, filename=filename, file_path = file_path, pred = pred)   
        db.session.add(emo)
        db.session.commit()


        return render_template('predict.html', pred = pred, user_image = file_path)
    
if __name__ == "__main__":
    app.run(debug=True) 
    
    
    
    
    
    
    
    