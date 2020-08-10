import keras
import numpy as np
import librosa
import time
import sys
from WavFileHelper import WavFileHelper

MODEL_PATH = "currentModel.h5"
NUM_SAMPLES_TO_CONCIDER = 44100


class _EmotionSpottingService:
    model = None
    _mappings = ["nothappy", "happy", "interested"]
    _instance = None

    def predict(self, filePath):
        helper = WavFileHelper()
        MFCCs = helper.extract_mfcc(filePath)
        if len(MFCCs) == 0:
            print('Audio snippet is too short!')
            return 'FAILURE'
        emotionPredictions = []
        seconds = 0
        for mfcc in MFCCs:
            reshapedMfcc = mfcc[np.newaxis, ..., np.newaxis]
            predictions = self.model.predict(reshapedMfcc)
            predictedIndex = np.argmax(predictions)
            predictedKeyword = self._mappings[predictedIndex]
            timeCode = time.strftime('%H:%M:%S', time.gmtime(seconds))
            emotionPredictions.append(
                {"timeCode": timeCode, "emotion": predictedKeyword})
            seconds += 2
        return emotionPredictions


def EmotionSpottingService():
    if _EmotionSpottingService._instance == None:
        _EmotionSpottingService._instance = _EmotionSpottingService()
        _EmotionSpottingService.model = keras.models.load_model(MODEL_PATH)
    return _EmotionSpottingService._instance
