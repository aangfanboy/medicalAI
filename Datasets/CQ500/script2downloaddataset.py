import os
import sys
import wget
import zipfile


def bar_progress(current, total, _):
    progress_message = "Downloading: %d%% [%d / %d] MB" % (current / total * 100, current/1048576, total/1048576)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def downloader(files_txt_path: str = "cq500_files.txt", download_first_many: int = -1):
    os.makedirs("all_data", exist_ok=True)

    file = open(files_txt_path)

    lines = file.readlines()
    prediction_probabilities_path = lines[-3]
    reads_path = lines[-4]

    wget.download(prediction_probabilities_path, out="all_data/")
    wget.download(reads_path, out="all_data/")

    for i, line in enumerate(lines[:download_first_many]):
        print(f"Downloading --> {line} ||| [{i}/{len(lines)}]")
        wget.download(line, bar=bar_progress, out="all_data/")

    file.close()


def unzip():
    for file in os.listdir("all_data"):
        if file.endswith(".zip"):
            file = os.path.join("all_data", file)
            print(f"unzipping {file}")

            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(file.replace(".zip", ""))

            print(f"deleting {file}")

            os.remove(file)


if __name__ == '__main__':
    downloader(download_first_many=40)
    unzip()
