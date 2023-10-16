import numpy as np
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub
import PIL.Image
import cv2

import streamlit as st



class utils:
    
    model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    
    logo_url = "https://raw.githubusercontent.com/dorukcanga/Style-Transfer-Streamlit-App/main/tofi_vangogh.jpg"

    style_folder = './style_images/'

    default_style_dict = {
        'Van Gogh - The Starry Night' : style_folder+'van_gogh_starry_night.jpg',
        'Van Gogh - Self Portrait' : style_folder+'van_gogh_self_portrait.jpeg',
        'Monet - Impression, Sunrise' : style_folder+'monet_impression_sunrise.jpeg',
        'Da Vinci - Vitruvian Man' : style_folder+'davinci_vitruvian_man.jpeg',
        'Klimt - The Kiss' : style_folder+'klimt_the_kiss.jpeg',
        'Picasso - Two Girls Reading' : style_folder+'picasoo_two_girls_reading.jpeg',
        'Picasso - Guernica' : style_folder+'picasso_guernica.jpeg',
        'Hokusai - The Great Wave off Kanagawa' : style_folder+'hokusai_great_wave_off_kanagawa.jpeg',
        'Munch - The Scream' : style_folder+'munch_the_scream.jpeg',
        'Other - Candy' : style_folder+'other_candy.jpeg',
        'Other - Mosaic' : style_folder+'other_mosaic.jpeg'
    }

@st.cache_resource
def load_hub_model(url):
    
    model = hub.load(url)
    return model


def load_image(path, size=256):
    
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=(size, size)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = image[tf.newaxis, :]
    
    return image


def tensor_to_image(tensor):
    
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def preprocess_image(image):
        image2= PIL.Image.open(image)
        image2 = np.array(image2)
        image2 = cv2.resize(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), (512, 512))
        image2 = image2 / 255.0
        image2 = tf.image.convert_image_dtype(image2, tf.float32)[tf.newaxis, :]
        
        return image2

