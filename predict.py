import tensorflow as tf

model = tf.keras.models.load_model("cat_dog_classify.keras")

image_path = "sample1.jpg"  
image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

predictions = model.predict(image_array)
if predictions[0][0] > 0.5:
    print("The image is classified as a Dog.")
    print("Prediction confidence: ", predictions[0][0]*100,"%")
else:
    print("The image is classified as a Cat.")
    print("Prediction confidence: ", (1 - predictions[0][0])*100,"%")