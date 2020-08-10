import os
import EmotionSpottingService
import math
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/ededore-archive"
app.config["SECRET_KEY"] = os.urandom(20)
mongo = PyMongo(app)
predictor = EmotionSpottingService.EmotionSpottingService()


@app.route('/', methods=['GET', 'POST'])
def uploadAudio():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # try to make predictions
            prediction = predictor.predict(file)
            if prediction == "FAILURE":
                flash(
                    "Couldn't make any predictions, audio must have sample rate 441000 and be at least 2 sec long.")
                return redirect(request.url)
            countDict = {}
            for dataset in prediction:
                key = dataset["emotion"]
                if not key in countDict:
                    countDict[key] = 1
                else:
                    countDict[key] = countDict[key] + 1
            mostOccuredEmotion = list(countDict.keys())[0]
            for key in countDict.keys():
                emotionCount = countDict[key]
                if emotionCount > countDict[mostOccuredEmotion]:
                    mostOccuredEmotion = key
            q25Index = math.floor(len(prediction)/4)
            q50Index = math.floor(len(prediction)/2)
            q75Index = math.floor(len(prediction)/4)*3
            # save data
            file_id = mongo.save_file(filename, file)
            archive_data = {"emotions": prediction,
                            "timeCodeCount": len(prediction),
                            "mostOccuredEmotion": mostOccuredEmotion,
                            "emotionCount": countDict,
                            "oneFourthQuantile": prediction[q25Index],
                            "oneHalfQuantile": prediction[q50Index],
                            "threeFourthsQuantile": prediction[q75Index],
                            "audio": file_id,
                            "personAlias": request.form["alias"],
                            "description": request.form["description"],
                            "authenticity": request.form["authenticity"],
                            "title": request.form["title"]}
            inserted_id = mongo.db.data.insert_one(archive_data).inserted_id
            flash('Data successfully archived under id '+str(inserted_id))
            return redirect(request.url)
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
