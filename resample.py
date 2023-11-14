import os
import pandas as pd


if __name__ == "__main__":
    spec_path = "E:\pythor/clips"
    label_path = "E:\pythor\labels.csv"

    df = pd.read_csv(label_path, sep=';', usecols=['path'])
    validated = df.path
    for path, _, fnames in os.walk(spec_path):
        for file in fnames:
            # for x in paths:
            if file not in validated.values:
                print(file)  # (256, 862)
                file_path = os.path.join(spec_path, file)
                os.remove(file_path)
                # spectogram = np.load(file_path)
                # database.append(spectogram)
