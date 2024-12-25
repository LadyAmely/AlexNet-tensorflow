import os
import sys
models_dir = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(models_dir)
from models.alexnet import AlexNet
from dataset.dataset_loader import load_and_preprocess_data
import tensorflow as tf

def train_model():
    train_ds, test_ds = load_and_preprocess_data(batch_size=32, train_subset_size=1000, test_subset_size=200,
                                                 increase_factor=2)

    model = AlexNet(input_shape=(224, 224, 3), num_classes=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    model.fit(train_ds, validation_data=test_ds, epochs=50, callbacks=[early_stopping, lr_scheduler])
    model.save('../saved_models/alexnet.keras')
    model.export('../saved_models/alexnet')
    model.save('../saved_models/alexnet.h5')


if __name__ == "__main__":
    train_model()

