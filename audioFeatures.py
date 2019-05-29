from typing import List, Any, Union
import pyAudioAnalysis
import numpy
import os
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler

cols = [
    'name',
    'zcr',
    'energy',
    'eoe',
    'sCentroid',
    'sEntropy',
    'sRolloff'
]


def extractFromSoundClip(F):
    zero_crossing_rate_Delta = max(F[0]) - min(F[0])
    energy_Delta = max(F[1]) - min(F[1])
    entropy_of_energy_Delta = max(F[2]) - min(F[2])
    spectral_centroid_Delta = max(F[3]) - min(F[3])
    spectral_entropy_Delta = max(F[5]) - min(F[5])
    spectral_rolloff_Delta = max(F[7]) - min(F[7])
    # List of 6 features extracted from sound for EVERY AUDIO CLIP
    sound_features = [
        zero_crossing_rate_Delta,
        energy_Delta,
        entropy_of_energy_Delta,
        spectral_centroid_Delta,
        spectral_entropy_Delta,
        spectral_rolloff_Delta
    ]
    return sound_features


FULL_VIDEO_DIR = 'video_folder_path'
SCENE_AUDIO_DIR = 'scene_audios_path'

fullVideos = os.listdir(SCENE_AUDIO_DIR)
allAudios = sorted(os.listdir(SCENE_AUDIO_DIR))

fault_count = 0
ALL_FEATURES = []

for fullVid in fullVideos:

    currentAudios = []
    for aud in allAudios:
        if aud.find(fullVid[:-4]) >= 0:
            currentAudios.append(aud)

    ALL_SOUND_FEATURES = []
    removalList = []
    for aud in currentAudios:
        try:
            [Fs, x] = audioBasicIO.readAudioFile(SCENE_AUDIO_DIR + aud)
            F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:, 0], Fs, 0.050 * Fs, 0.025 * Fs)
            soundFeatures = extractFromSoundClip(F)  # list of 6 numerical features for that clip
            # print(soundFeatures)
            soundFeatures = [0 if math.isnan(x) else x for x in soundFeatures]
            ALL_SOUND_FEATURES.append(soundFeatures)
        except:
            removalList.append(aud)
            print('fault in', aud)
            fault_count = fault_count + 1

    if (len(ALL_SOUND_FEATURES) == 0):
        continue
    currentAudios = sorted(list(set(currentAudios) - set(removalList)))
    scaler = MinMaxScaler()
    print(fullVid)
    scaler.fit(ALL_SOUND_FEATURES)
    ALL_SOUND_FEATURES_SCALED = scaler.transform(ALL_SOUND_FEATURES)
    ALL_SOUND_FEATURES_SCALED = (ALL_SOUND_FEATURES_SCALED.tolist())
    if len(ALL_SOUND_FEATURES_SCALED) != len(currentAudios):
        print("Exception in lengths of input samples and feature vectors")
        exit(1)

    FULL_VID_FEATURES = []  # type: List[Union[List[str], Any]]
    for i in range(len(currentAudios)):
        feature_row = [currentAudios[i]] + ALL_SOUND_FEATURES_SCALED[i]
        FULL_VID_FEATURES.append(feature_row)

    ALL_FEATURES = ALL_FEATURES + FULL_VID_FEATURES

# print('ALL FEATURES LENGTH = ', len(ALL_FEATURES))
MEGA_FRAME = pd.DataFrame(ALL_FEATURES, columns=cols)

OUTPUT_CSV = '/Users/sumanur/Desktop/adplace/sound_features_v1.csv'
MEGA_FRAME.to_csv(OUTPUT_CSV)

# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
