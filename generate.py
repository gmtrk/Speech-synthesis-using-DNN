import librosa
import os
import numpy as np
import soundfile as sf
import procces
from keras import models
import string


def convert_spectogram_to_audio(spectogram):
    log_spectogram = spectogram[:, :, 0]
    spectogram_in_amplitude = librosa.db_to_amplitude(log_spectogram)
    signal = librosa.istft(spectogram_in_amplitude, hop_length=256)
    print(signal.shape)
    return signal


def one_hot_encoding(sample):
    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    token_index.update({"ą": 101,
                        "ć": 102,
                        "ę": 103,
                        "ł": 104,
                        "ń": 105,
                        "ó": 106,
                        "ś": 107,
                        "ż": 108,
                        "ź": 109,
                        "Ą": 110,
                        "Ć": 111,
                        "Ę": 112,
                        "Ł": 113,
                        "Ń": 114,
                        "Ó": 115,
                        "Ś": 116,
                        "Ż": 117,
                        "Ź": 118,
                        })
    max_len = 150
    result = np.zeros((max_len, max(token_index.values()) + 1))
    for j, characters in enumerate(sample[:max_len]):
        index = token_index.get(characters)
        result[j, index] = 1
    return result


if __name__ == "__main__":
    HOP_LENGHT = 256
    synthesizer = models.load_model('E:\pythor/newspeechmodel')
    sentence = "Mówili będą z ciebie ludzie"
    embedded_sentence = one_hot_encoding(sentence)
    embedded_sentence = embedded_sentence[None, :]
    #print(embedded_sentence)
    spectogram = synthesizer.predict(embedded_sentence)

    spectogram = spectogram * 100000000000000
    print(spectogram.max())
    print(spectogram.min())
    signal = convert_spectogram_to_audio(spectogram[0,:,:,:])
    #print(spectogram.shape)
    #print(sf.read('E:\pythor\speech/1.wav')[0])



    save_dir = "E:\pythor\SAVED"
    #print(signal)
    save_path = os.path.join(save_dir, "test22.wav")
    sf.write(save_path, signal, 8000)
