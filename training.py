import tensorflow as tf
from tensorflow.keras import layers

train_ds = tf.keras.utils.image_dataset_from_directory("dataset/train", image_size=(160, 160),label_mode='binary')
testing_ds = tf.keras.utils.image_dataset_from_directory("dataset/validation", image_size=(160, 160),label_mode='binary')

base= tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base.trainable = False

model=tf.keras.Sequential([layers.Rescaling(1./255,input_shape=(160,160,3)),base,layers.GlobalAveragePooling2D(),layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ds, validation_data=testing_ds, epochs=5)

model.save("cat_dog_classify.keras")
print("Model saved successfully.")