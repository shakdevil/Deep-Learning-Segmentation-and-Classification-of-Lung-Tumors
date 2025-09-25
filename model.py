import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

class SegmentationModel():
    def __init__(self, model_path, input_shape=(256, 256, 3), num_classes=3):
        self.model = tf.keras.models.load_model(model_path)   


    def build_model(self):
        """Defines a U-Net++ architecture for segmentation."""
        inputs = layers.Input(self.input_shape)

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        # Decoder
        u1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(p1)
        u1 = layers.concatenate([u1, c1])
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)

        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(c2)
        model = models.Model(inputs, outputs)
        return model

    def preprocess_image(self, image):
        """Preprocess an image before feeding it into the model."""
        image = image.resize((self.input_shape[0], self.input_shape[1]))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array

    def predict_segmentation(self, image):
        """Generate the segmentation mask for an input image."""
        img_array = self.preprocess_image(image)
        prediction = self.model.predict(img_array)[0] 
        return np.argmax(prediction, axis=-1)

    def overlay_contours(self, image, mask):
        """Draw segmentation contours on the original image."""
        image = np.array(image)
        mask = np.uint8(mask) 
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 2)
        return overlay
