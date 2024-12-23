import tensorflow as tf

class AlexNet():

    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.conv1_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='SAME', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='SAME', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv3_1 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv3_2 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):

        x = self.conv1_1(inputs)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        outputs = self.output_layer(x)

        return outputs

input_tensor = tf.keras.Input(shape=(224, 224, 3))
model_instance = AlexNet(num_classes=1000)
outputs = model_instance.call(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

model.summary()
