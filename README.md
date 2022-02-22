# teknofestcomp
https://www.teknofest.org/tr/competitions/competition/34


# [Veri setleri](/Datasets)
* ###[CQ500](/Datasets/CQ500)
  * qure.ai websitesinden alındı
  * [script2downloaddataset.py](Datasets/CQ500/script2downloaddataset.py) ile veri seti indirilebilir, model geliştirirken datasetin hepsini indirmenize gerek yok ilk 30-40 zip dosyası yeterli
  * [displayexample.py](Datasets/CQ500/displayexample.py) ile örnek bi resmi display edip etiketlerine bakabilirsiniz. Etiketler probability şeklinde, yuvarlayarak 1 ve 0 değerlerine ulaşabilirsiniz
  * Modelin örnek sonuçları için [buraya](http://headctstudy.qure.ai/explore_data) bakabilirsiniz
  * 



# Classification için to-do
- [] dataseti stream eden bi class yazılmalı. klasik tf.Dataset formatı, output olarak resim(x) ve ICH(y)(ICH'yi 1 ve 0'a yuvarlayarak) değerini verecek
- [] CNN modeli oluşturulmalı. dataset olarak üstteki tf.Dataset objesini kullanacak, son layer'ı Dense(2, activation="softmax") şeklinde olmalı. Tensorboard kullanılması + olur)


# Yarışma için to-do
- [] teknofest raporu için çekirdek yazımı