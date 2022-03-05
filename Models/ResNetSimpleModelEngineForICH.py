import os
import tensorflow as tf

from Datasets import DatasetEngine
from TensorBoardEngine import TensorBoardCallback


class SimpleModelEngine:
    @tf.function
    def test_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.loss_function(y, logits)

        return logits, tf.reduce_mean(loss)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_function(y, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return logits, tf.reduce_mean(loss)

    def __init__(self, epochs: int = 5):
        self.data_engine = DatasetEngine.DataEngineTFRecord(
            "../Datasets/CQ500_all_data.tfrecord",
            batch_size=8,
            epochs=1,  # set to -1 so it can stream forever
            buffer_size=30000,
            reshuffle_each_iteration=False,
            test_batch=20,
            map_to=True,
            image_raw_tfrecord_label="image_raw",
            labels_tfrecord_label=["ich"],  # only streams 'ich' value, more labels in Datasets/CQ500/maketfrecord.py
            function_for_image_tfrecord=lambda x: tf.reshape(x, (112, 112, 1)),
            function_for_labels_tfrecord=lambda x: tf.cond(x < 0.5, true_fn=lambda: 0, false_fn=lambda: 1)
        )
        load_from_model_path = True
        self.model_path = "resnet50.h5"

        if load_from_model_path and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = tf.keras.applications.ResNet50(input_shape=(112, 112, 1), classes=2, weights=None)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.sparse_categorical_crossentropy
        # self.steps_per_epoch = sum([1 for _ in self.data_engine.dataset])
        self.steps_per_epoch = 21321
        self.epochs = epochs
        self.tensorboard_engine = TensorBoardCallback(logdir="classifier_tensorboard")

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), (-1, 1))
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    def train_model(self, print_in_evey_n: int = 1):
        acc_mean = tf.keras.metrics.Mean()
        loss_mean = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(self.data_engine.dataset):
            logits, loss = self.train_step(x, y)
            acc = self.calculate_accuracy(y, logits)

            loss_mean(loss)
            acc_mean(acc)

            if i % print_in_evey_n == 0:
                print(f"[{i}/{self.steps_per_epoch}] Loss --> {loss_mean.result().numpy()} "
                      f"|| Acc --> {acc_mean.result().numpy()}")

            self.tensorboard_engine({"loss": loss, "accuracy": acc})

            if i % 100 == 0:
                loss_mean.reset_states()
                acc_mean.reset_states()

    def test_model(self, epoch: int = -1):
        acc_mean = tf.keras.metrics.Mean()
        loss_mean = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(self.data_engine.dataset_test):
            logits, loss = self.test_step(x, y)
            acc = self.calculate_accuracy(y, logits)

            loss_mean(loss)
            acc_mean(acc)

            if epoch != -1:
                self.tensorboard_engine.add_with_step({"loss_test": loss, "accuracy_test": acc}, step=epoch)

        print(f"Loss Val. --> {loss_mean.result().numpy()}  || Acc Val. --> {acc_mean.result().numpy()}")

    def save_model(self):
        tf.keras.models.save_model(self.model, self.model_path)

        print(f"model saved to {self.model_path}")

    def __call__(self, *args, **kwargs):
        self.tensorboard_engine.initialize(
            delete_if_exists=True
        )

        for epoch in range(self.epochs):
            print(f"running epoch [{epoch+1}/{self.epochs}]")

            self.train_model(print_in_evey_n=100)
            self.test_model(epoch=epoch)
            self.save_model()
            print("--------------------------")


if __name__ == '__main__':
    sme = SimpleModelEngine()
    sme()
