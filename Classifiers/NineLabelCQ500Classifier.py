import tensorflow as tf

from MainModelEngine import MainModelEngine


class L9ModelEngine(MainModelEngine):
    def load_model(self, num_classes: int = 2, input_shape=(112, 112, 1)):
        if self.load_from_model_path and tf.io.gfile.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
        else:
            model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights=None)

            x = model.layers[-1].output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            x1 = tf.keras.layers.Dense(1, activation='sigmoid', name='ich_prob')(x)
            x2 = tf.keras.layers.Dense(1, activation='sigmoid', name='iph_prob')(x)
            x3 = tf.keras.layers.Dense(1, activation='sigmoid', name='ivh_prob')(x)
            x4 = tf.keras.layers.Dense(1, activation='sigmoid', name='sdh_prob')(x)
            x5 = tf.keras.layers.Dense(1, activation='sigmoid', name='edh_prob')(x)
            x6 = tf.keras.layers.Dense(1, activation='sigmoid', name='sah_prob')(x)
            x7 = tf.keras.layers.Dense(1, activation='sigmoid', name='calvarial_fracture_prob')(x)
            x8 = tf.keras.layers.Dense(1, activation='sigmoid', name='mass_effect_prob')(x)
            x9 = tf.keras.layers.Dense(1, activation='sigmoid', name='midline_shift_prob')(x)

            model = tf.keras.models.Model(model.layers[0].input, [x1, x2, x3, x4, x5, x6, x7, x8, x9])

        model.summary()

        return model

    def __init__(self):
        super().__init__(
            tf_record_path = "../Datasets/CQ500/CQ500_all_data.tfrecord",
            num_of_images = 170568,
            labels_tfrecord_label=["ich", "iph", "ivh", "sdh", "edh", "sah",
                                   "calvarial_fracture", "mass_effect", "midline_shift"],
            model_path="resnet50PL.h5",
            batch_size=128,
            lr_dict={0:0.001, 1:0.0005, 3: 0.0001}
        )


if __name__ == "__main__":
    l9me = L9ModelEngine()
    l9me()
