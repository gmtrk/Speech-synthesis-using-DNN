import os
import ffmpeg
from pydub import AudioSegment

if __name__ == "__main__":
    # files
    spec_path = "E:\pythor/clips"
    save_path = "E:\pythor\clippers"
    # convert wav to mp3

    for path, _, fnames in os.walk(spec_path):
        for file in fnames:
            # for x in paths:
              # (256, 862)
            file_path = os.path.join(spec_path, file)
            print(file_path)
            sound = AudioSegment.from_mp3(file_path)
            save_pathu = os.path.join(save_path, file)
            sound.export(save_pathu + ".wav", format="wav")
            # spectogram = np.load(file_path)
            # database.append(spectogram)