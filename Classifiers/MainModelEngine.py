import sys
import tensorflow as tf

sys.path.append("../")

from Datasets.DatasetEngine import DataEngineTFRecord
from ModelUtils.TensorBoardEngine import TensorBoardCallback


class MainModelEngine:
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

    def load_model(self, num_classes: int = 2, input_shape=(112, 112, 1)):
        if self.load_from_model_path and os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
        else:
            model = tf.keras.applications.ResNet50(input_shape=input_shape, classes=num_classes, weights=None)

        return model

    def weighted_binary_ce_multiple(self, y_true, y_pred):
      y_pred = tf.concat([_ for _ in y_pred], axis=-1)
      y_true = tf.cast(y_true, tf.float32)

      loss = 0
      for i in range(y_true.shape[1]):
        true11 = y_true[:, i:(i+1)]
        pred11 = y_pred[:, i:(i+1)]
        bce_loss11 = tf.keras.losses.binary_crossentropy(true11, pred11, axis=-1)
        weights11 = true11 * self.one_weight[i] + (1. - true11) * (1-self.one_weight[i])

        loss11 = weights11*bce_loss11
        loss += tf.reduce_mean(loss11)

      return loss
  
    def weighted_binary_ce(self, y_true, y_pred):
      y_true = tf.cast(y_true, tf.float32)
      bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
      weights = y_true * self.one_weight + (1. - y_true) * (1-self.one_weight)

      return tf.reduce_mean(weights * bce_loss)

    def check_for_lr(self, which_epoch_this_is: int):
      if self.lr_dict is not None:
        for epoch in self.lr_dict:
          lr_related = self.lr_dict[epoch]
          if which_epoch_this_is == epoch:
            self.optimizer.learning_rate = lr_related

        assert self.optimizer.learning_rate == self.optimizer.lr

    def __init__(self, tf_record_path: str, epochs: int = 5, batch_size: int = 8, num_of_images: int = 0,
                 labels_tfrecord_label=None, model_path: str = "resnet50.h5",
                 lr_dict = None):
        if labels_tfrecord_label is None:
            labels_tfrecord_label = ["ich"]

        self.batch_size = batch_size
        self.lr_dict = lr_dict
        self.data_engine = DataEngineTFRecord(
            tf_record_path,
            batch_size=self.batch_size,
            epochs=1,  # set to -1 so it can stream forever
            buffer_size=10000,
            reshuffle_each_iteration=False,
            test_batch=int((20*8)/self.batch_size),
            map_to=True,
            image_raw_tfrecord_label="image_raw",
            labels_tfrecord_label=labels_tfrecord_label,
            function_for_image_tfrecord=lambda x: tf.reshape(x, (112, 112, 1)),
            function_for_labels_tfrecord=lambda x: tf.cond(x < 0.5, true_fn=lambda: 0, false_fn=lambda: 1),
        )
        self.load_from_model_path = True
        self.write_loss_step = False
        self.model_path = model_path
        self.multiple_labels = True if len(labels_tfrecord_label) > 1 else False

        self.one_weight = [0.59, 0.69, 0.93, 0.91, 0.98, 0.88, 0.84, 0.79, 0.85]

        self.model = self.load_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        if self.multiple_labels:
          self.loss_function = self.weighted_binary_ce_multiple
        else:
          self.loss_function = self.weighted_binary_ce

        if num_of_images == 0:
            self.steps_per_epoch = sum([1 for _ in self.data_engine.dataset])
        else:
            self.steps_per_epoch = int((num_of_images/self.batch_size))

        self.epochs = epochs
        self.tensorboard_engine = TensorBoardCallback(logdir="classifier_tensorboard")

    def calculate_accuracy(self, y_true, y_pred):
        if self.multiple_labels:
          y_pred = tf.concat([_ for _ in y_pred], axis=-1)

        return tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(y_pred), tf.int32), tf.cast(tf.round(y_true), tf.int32)), tf.float32))

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
                
                self.write_loss_step = True

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

            self.check_for_lr(epoch)
            self.train_model(print_in_evey_n=100)
            self.test_model(epoch=epoch)
            self.save_model()
            print("--------------------------")
           
