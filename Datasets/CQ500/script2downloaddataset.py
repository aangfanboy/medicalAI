import os
import wget
import time
import queue
import zipfile
import threading

from tqdm import tqdm


class MainThreadNumber:
    KILLED_DOWNLOADERS = 0
    KILLED_UNZIPPERS = 0
    DOWNLOADERS_THREAD = 12
    UNZIPPERS_THREAD = 4

    def unzipper(self):
        def while_zip_qsize():
            while self.zip_files2unzip.qsize() > 0:
                path = self.zip_files2unzip.get()
                print(f"\nunzipping {path}")

                try:
                    with zipfile.ZipFile(path, "r") as zip_ref:
                        zip_ref.extractall(path.replace(".zip", ""))
                except:
                    print(f"an error occurred in {path}, deleting zip file without unzipping")
                finally:
                    os.remove(path)
                    print(f"\nunzipped {path}")
                    self.tqdm_bar.update()

        while not self.links2download_as_zip.empty() or self.KILLED_DOWNLOADERS != self.DOWNLOADERS_THREAD:
            while_zip_qsize()
            time.sleep(0.5)

        time.sleep(3)
        while_zip_qsize()

        self.KILLED_UNZIPPERS += 1
        print("UNZIPPER KILLED")

    def downloader(self):
        while not self.links2download_as_zip.empty():
            link = self.links2download_as_zip.get()
            print(f"\nDownloading {link}")
            path = wget.download(link, out="all_data/")
            self.zip_files2unzip.put(path)
            print(f"\nDownloaded {link}")

        self.KILLED_DOWNLOADERS += 1
        print("DOWNLOADER KILLED")

    def __init__(self, files_txt_path: str = "cq500_files.txt", download_first_many: int = -1):
        os.makedirs("all_data", exist_ok=True)

        file = open(files_txt_path)

        lines = file.readlines()
        prediction_probabilities_path = lines[-3]
        reads_path = lines[-4]

        wget.download(prediction_probabilities_path, out="all_data/")
        wget.download(reads_path, out="all_data/")

        self.links2download_as_zip = queue.Queue()

        for i, line in enumerate(lines[:download_first_many]):
            line = line.strip()
            if line[-3:] == "zip":
                self.links2download_as_zip.put(line)

        self.zip_files2unzip = queue.Queue()
        self.tqdm_bar = tqdm(total=len(lines[:download_first_many]), desc="Downloaded and unzipped")

    def __call__(self, *args, **kwargs):
        for i in range(self.DOWNLOADERS_THREAD):
            worker = threading.Thread(target=self.downloader, daemon=True)
            worker.start()

        for i in range(self.UNZIPPERS_THREAD):
            worker = threading.Thread(target=self.unzipper, daemon=True)
            worker.start()

        while not self.links2download_as_zip.empty():
            time.sleep(1)

        while not self.zip_files2unzip.empty():
            time.sleep(1)

        while self.KILLED_DOWNLOADERS != self.DOWNLOADERS_THREAD:
            time.sleep(1)

        while self.KILLED_UNZIPPERS != self.UNZIPPERS_THREAD:
            time.sleep(1)

        self.tqdm_bar.close()
        print("ended")


if __name__ == '__main__':
    mtn = MainThreadNumber(download_first_many=-1)  # 11.36
    mtn()
