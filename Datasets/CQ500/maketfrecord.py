import glob
import pandas as pd
import tensorflow as tf
from pydicom import dcmread

from tqdm import tqdm


class TFRecordConverter:
    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def image_reader(path):
        image = tf.expand_dims(dcmread(path).pixel_array, axis=-1)
        image = tf.image.resize(image, (112, 112), method="nearest")
        image = tf.image.random_flip_left_right(image)

        return tf.cast(image, tf.float32) / 255.

    def __init__(self, tfrecord_path: str = "CQ500_all_data.tfrecord"):
        self.tfrecord_path = tfrecord_path
        self.all_file_paths = glob.glob("all_data/*/*/*/*/*.dcm")
        self.pred_prob_csv = pd.read_csv("all_data/prediction_probabilities.csv")

        self.writer = tf.io.TFRecordWriter(self.tfrecord_path)

    def __call__(self, *args, **kwargs):

        for path in tqdm(self.all_file_paths, "adding data to TFRecord file"):
            name, category_new, ich, iph, ivh, sdh, edh, sah, calvarial_fracture, mass_effect, midline_shift = \
                self.pred_prob_csv[self.pred_prob_csv["name"] == path.split("\\")[1]].values[0]

            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'name': self._bytes_feature(bytes(name.encode("utf-8"))),
                        'category_new': self._bytes_feature(bytes(category_new.encode("utf-8"))),
                        'ich': self._float_feature(ich),
                        'iph': self._float_feature(iph),
                        'ivh': self._float_feature(ivh),
                        'sdh': self._float_feature(sdh),
                        'edh': self._float_feature(edh),
                        'sah': self._float_feature(sah),
                        'calvarial_fracture': self._float_feature(calvarial_fracture),
                        'mass_effect': self._float_feature(mass_effect),
                        'midline_shift': self._float_feature(midline_shift),
                        'image_raw': self._bytes_feature(bytes(self.image_reader(path).numpy())),
                    }
                )
            )

            self.writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    tf_rc = TFRecordConverter()
    tf_rc()
