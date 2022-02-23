import tensorflow as tf


class DataEngineTFRecord:
    def mapper(self, tfrecord_data):
        features = {
            self.image_raw_tfrecord_label: tf.io.FixedLenFeature([], tf.string),
        }

        for qq in self.labels_tfrecord_label:
            features[qq] = tf.io.FixedLenFeature([], tf.float32)

        features = tf.io.parse_single_example(tfrecord_data, features)

        image = self.function_for_image_tfrecord(tf.io.decode_raw(features[self.image_raw_tfrecord_label], tf.float32))
        labels = [self.function_for_labels_tfrecord(features[qq]) for qq in self.labels_tfrecord_label]

        return image, labels

    def __init__(self, tf_record_path: str, batch_size: int = 16, epochs: int = 10, buffer_size: int = 50000,
                 reshuffle_each_iteration: bool = True, test_batch=64, map_to: bool = True,
                 image_raw_tfrecord_label: str = "image_raw", labels_tfrecord_label=None,
                 function_for_image_tfrecord=lambda x: x, function_for_labels_tfrecord=lambda x: x):

        if labels_tfrecord_label is None:
            labels_tfrecord_label = []

        self.image_raw_tfrecord_label = image_raw_tfrecord_label
        self.labels_tfrecord_label = labels_tfrecord_label
        self.function_for_image_tfrecord = lambda x: function_for_image_tfrecord(x)
        self.function_for_labels_tfrecord = lambda x: function_for_labels_tfrecord(x)

        self.dataset_test = None
        if test_batch > 0:
            reshuffle_each_iteration = False
            print(f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if "
                  f"tf.data will fixed.")

        self.tf_record_path = tf_record_path

        self.dataset = tf.data.TFRecordDataset(self.tf_record_path)
        if buffer_size > 0:
            self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)
            print("[*] shuffled")

        if map_to:
            self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            print("[*] mapped")
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
        print("[*] batched")

        if test_batch > 0:
            self.dataset_test = self.dataset.take(int(test_batch))
            self.dataset = self.dataset.skip(int(test_batch))
            print("[*] test batch split")

        self.dataset = self.dataset.repeat(epochs)


if __name__ == '__main__':
    data_engine = DataEngineTFRecord(
        "CQ500/CQ500_all_data.tfrecord",
        batch_size=8,
        epochs=1,  # set to -1 so it can stream forever
        buffer_size=500000,
        reshuffle_each_iteration=True,
        test_batch=5,
        map_to=True,
        image_raw_tfrecord_label="image_raw",
        labels_tfrecord_label=["ich"],  # only streams 'ich' value, more labels in Datasets/CQ500/maketfrecord.py
        function_for_image_tfrecord=lambda x: tf.reshape(x, (112, 112, 1)),
        function_for_labels_tfrecord=lambda x: tf.cond(x < 0.5, true_fn=lambda: 0, false_fn=lambda: 1)
    )
