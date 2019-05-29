AUDIO_DIR = '/Users/sumanur/GoogleDrive/adplace/scene_audios/'
# AUDIO_FILE = AUDIO_DIR + 'audio-adplace_001-Scene-029.wav'

# !/usr/bin/env python3

import speech_recognition as sr
import json
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from os import path

def convertSpeechToText(AUDIO_FILE):
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    GOOGLE_CLOUD_SPEECH_CREDENTIALS = { # Google api credentials json here
        "key1" : "val1",
        "key2" : "val2"
    }

    GOOGLE_CLOUD_SPEECH_CREDENTIALS = json.dumps(GOOGLE_CLOUD_SPEECH_CREDENTIALS)

    try:
        ans = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
        return ans
    except sr.UnknownValueError:
        print("Google Cloud Speech could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Cloud Speech service; {0}".format(e))
        return ""
    except:
        return ""

# Returns absolute sentiment score by vader sentiment analysys
def vaderify(sentence):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(sentence)
    return abs(scores.get('compound'))

SOUND_FEATURES_CSV = '/Users/sumanur/Desktop/adplace/final_frame_test.csv'
frame = pd.read_csv(SOUND_FEATURES_CSV)

frame = frame.drop(columns=['Unnamed: 0'])

for index, row in frame.iterrows():
    AUDIO_FILE = AUDIO_DIR + row['name']
    reply = convertSpeechToText(AUDIO_FILE)
    score = vaderify(reply)
    row['sentiment'] = score

OUTPUT_CSV = '/Users/sumanur/Desktop/adplace/sentiment_features.csv'
frame.to_csv(OUTPUT_CSV)