import os
import zipfile


def download_and_extract_data():
    os.system("wget --no-check-certificate 'https://cloud.ml.jku.at/s/CdYdidkkBpFgcED/download' -O train_mixed.zip")
    # select as a data root the mixed demonstratoins directory
    data_root = './data-mixed'
    with zipfile.ZipFile('train_mixed.zip', 'r') as zip_ref:
        os.makedirs(data_root, exist_ok=True)
        zip_ref.extractall(data_root)


if __name__ == "__main__":
    download_and_extract_data()
