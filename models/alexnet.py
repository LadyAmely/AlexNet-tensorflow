import tensorflow as tf
class AlexNet(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=11, strides=4, padding='SAME', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)


        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.fc2(x)
        if training:
            x = self.dropout2(x, training=training)
        outputs = self.output_layer(x)

        return outputs
