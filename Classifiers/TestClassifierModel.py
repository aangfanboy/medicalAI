import sys
import tensorflow as tf

sys.path.append("../")
from Datasets.DatasetEngine import DataEngineTFRecord


class Tester:
    def __init__(self, model_path: str, label_names: list):
        self.label_names = label_names
        self.model = tf.keras.models.load_model(model_path)

        self.input_shape = self.model.layers[0].input_shape[0][1:]


    def __call__(self, image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels = self.input_shape[-1])
        image = tf.image.resize(image, self.input_shape[:-1], method="nearest")
        image = tf.cast(image, tf.float32) / 255.
        image = tf.expand_dims(image, axis=0)

        outputs = self.model(image)
        outputs = tf.concat([_ for _ in outputs], axis=-1)
        outputs = tf.cast(tf.round(outputs), tf.int32)
        
        for label, output in zip(self.label_names, outputs[0]):
        	print(f"{label} ---> {output}")
    


if __name__ == "__main__":
    t = Tester("resnet50PLCQ500_9Label.h5", ["ich", "iph", "ivh", "sdh", "edh", "sah",
                                   "calvarial_fracture", "mass_effect", "midline_shift"])
    t("../TestImages/IHPCH.jpg")
