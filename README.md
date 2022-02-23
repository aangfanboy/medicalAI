# teknofestcomp
https://www.teknofest.org/tr/competitions/competition/34


# [Veri setleri](/Datasets)
* ### [CQ500](/Datasets/CQ500)
  * qure.ai websitesinden alındı
  * [script2downloaddataset.py](Datasets/CQ500/script2downloaddataset.py) ile veri seti indirilebilir, model geliştirirken datasetin hepsini indirmenize gerek yok ilk 30-40 zip dosyası yeterli
  * [displayexample.py](Datasets/CQ500/displayexample.py) ile örnek bi resmi display edip etiketlerine bakabilirsiniz. Etiketler probability şeklinde, yuvarlayarak 1 ve 0 değerlerine ulaşabilirsiniz
  * Modelin örnek sonuçları için [buraya](http://headctstudy.qure.ai/explore_data) bakabilirsiniz
  * 

# Classification için to-do

- [X] dataseti stream eden bi class yazılmalı. klasik tf.Dataset formatı, output olarak resim(x) ve ICH(y)(ICH'yi 1 ve 0'a yuvarlayarak) değerini verecek
- [ ] CNN modeli oluşturulmalı. dataset olarak üstteki tf.Dataset objesini kullanacak, son layer'ı Dense(2, activation="softmax") şeklinde olmalı. Tensorboard kullanılması + olur)

# Yarışma için to-do
- [ ] teknofest raporu için çekirdek yazımı


# Guideline

* Dataseti hazır hale getirmek için:
  1. [script2downloaddataset.py'yi](Datasets/CQ500/script2downloaddataset.py) kullanarak zip dosyalarını indirin ve unzip edin
  2. [displayexample.py](Datasets/CQ500/displayexample.py) ile verileri görebilirsiniz
  3. [maketfrecord.py](Datasets/CQ500/maketfrecord.py) ile TFRecord dosyasını oluşturun
  4. [DatasetEngine.py](Datasets/DatasetEngine.py) ile TFDataset formatında bir dataset oluşturun
  5. önceki adımdaki dataset objesini keras modeliniz ile kullanabilirsiniz. [Bakınız](https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras)
