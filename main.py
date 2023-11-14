import keras
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
from IPython.display import Audio

audio_fpath = "E:\pythor\speech"
labels_fpath = "E:\pythor\labels"
audio_clips = os.listdir(audio_fpath)
print("Liczba próbek w folderze speech = ",len(audio_clips))


s, sr = librosa.load(audio_fpath+ "/" + audio_clips[0], sr=None)

print(type(s), type(sr))
print(s.shape, sr)

plt.figure()
librosa.display.waveshow(s, sr=sr)
plt.show()

print(s.shape)


sgram = librosa.stft(s)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
plt.figure()
librosa.display.specshow(mel_scale_sgram, sr=sr, x_axis='time', y_axis='mel')
plt.show()
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
plt.figure()
librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
#librosa.display.specshow(mel_sgram)
plt.colorbar(format='%+2.0f dB')
plt.show()
final_mel=librosa.db_to_amplitude(mel_sgram)
audio = librosa.feature.inverse.mel_to_audio(final_mel)

print(audio)
write('test.wav',sr, audio)
