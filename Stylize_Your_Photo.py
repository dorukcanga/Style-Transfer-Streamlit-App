import numpy as np
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub
import PIL.Image
import cv2
from io import BytesIO

import streamlit as st

from utils import utils, load_hub_model, load_image, tensor_to_image, preprocess_image

st.set_page_config(layout="wide")

st.sidebar.title('Style Transfer Application')

logo_url = utils.logo_url
st.sidebar.image(logo_url)

st.title('Configurable Style Transfer Application with Multiple Styles')
st.header('Stylize Your Photo')



####################################################################################################
# Load Model from Tensorflow Hub
####################################################################################################

model = load_hub_model(utils.model_url)

####################################################################################################
# Content & Style Image Upload Sections
####################################################################################################

col1, col2= st.columns(2)

with col1:

    #Upload Content Image
    st.subheader('Upload Content Photo')
    
    tab1 = st.tabs(["Upload Content Image"])
    
    content_image = st.file_uploader("Please Upload JPG File", key=0)
    if content_image is not None:
        content_image2 = preprocess_image(content_image)

    
with col2:
    
    #Upload or Select Style Image
    st.subheader('Upload or Choose Style Photo(s)')
    
    tab1, tab2 = st.tabs(["Choose Style(s)", "Upload New Style(s)"])
    
    #Select Style Image Option
    selected_styles, uploaded_styles = [], []
    with tab1: 
        selected_styles = st.multiselect(
            'Choose From Existing Styles:',
             list(utils.default_style_dict.keys()))
    
    #Upload Style Image Option
    with tab2:
        uploaded_styles = st.file_uploader("Please Upload JPG File(s)", accept_multiple_files=True, key=1)
    if uploaded_styles is not None:
        uploaded_styles2=[]
        for file in uploaded_styles:
            file = preprocess_image(file)
            uploaded_styles2.append(file)
        
#Final Style Image List
selected_styles = selected_styles if type(selected_styles) == list else [selected_styles]
uploaded_styles = uploaded_styles if type(uploaded_styles) == list else [uploaded_styles]
num_styles = len(selected_styles) + len(uploaded_styles)

#Load Selected Styles
style_images = {}
if len(selected_styles) != 0:
    for i in selected_styles:
        temp_image = load_image(utils.default_style_dict[i], size=512)
        style_images[i] = temp_image

style_images2 = style_images.copy()
if len(uploaded_styles) != 0:
    for i in range(len(uploaded_styles)):
        style_images['uploaded_img'+str(i)] = uploaded_styles[i]
        style_images2['uploaded_img'+str(i)] = uploaded_styles2[i]
    

####################################################################################################
# Content & Style Image Printing Sections
####################################################################################################

col1, col2= st.columns(2)

with col1:
    st.subheader('Content Photo:')
    try:
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(content_image, caption='Content Image', use_column_width=True)
    except Exception:
        st.text("Content image\nwill be shown here\nonce uploaded.")

with col2:
    st.subheader('Style Photo(s):')
    try:
        if num_styles > 1:

            style_tabs = st.tabs(['Style - '+str(i+1) for i in range(num_styles)])
            for tab_no in range(num_styles):

                with style_tabs[tab_no]:

                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        temp_key = list(style_images.keys())[tab_no]
                        st.image(style_images[temp_key], caption='Style Image - '+str(tab_no+1), use_column_width=True)
        else:
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                temp_key = list(style_images.keys())[0]
                st.image(style_images[temp_key], caption='Style Image', use_column_width=True)
    except Exception:
        st.text("Style image(s)\nwill be shown here\nonce uploaded or choosen.")
                
                
####################################################################################################
# Content & Style Image Intensity Configuration
####################################################################################################                
                
st.subheader('Style & Original Content Intensity Configuration')

intensity_dict = {}

if num_styles == 0 or content_image == None:
    st.warning('Please upload content image and upload and/or select images to configurate intensity.')
else:
    st.info('Sum of intensity levels must be equal to 100%.')
    cols= st.columns(num_styles+1)
    for i in range(num_styles+1):
        if i == 0:
            with cols[i]:
                content_intensity = st.number_input('Content Intensity Level (%):', min_value=0, max_value=100, value=0, step=1)
                content_intensity = content_intensity / 100
                intensity_dict[i] = content_intensity
        else:
            with cols[i]:
                defualt_value = int(np.ceil(100/num_styles)) if i == 1 else int(np.floor(100/num_styles))
                temp_style_intensity = st.number_input('Style Intensity Level (%) - '+str(i)+':', min_value=0, max_value=100, value=defualt_value, step=1)
                temp_style_intensity = temp_style_intensity / 100
                intensity_dict[i] = temp_style_intensity

intensity_sum = np.ceil(np.sum(list(intensity_dict.values())))
if intensity_dict != {} and intensity_sum != 1:
    st.error('Sum of intensity levels must be equal to 100%.', icon="⚠️")
    
    
    
####################################################################################################
# Transfer Style(s)
####################################################################################################    

st.subheader('Stylize Your Photo')

exe = st.button("Run Model")

if exe:
    with st.spinner('Running...'):
        tensor_dict = {0 : content_image2}
        count=0
        for key, value in style_images2.items():
            count+=1
            tensor_dict[count] = model(tf.constant(content_image2), tf.constant(value))[0]


        stylized_tensor_final1 = 0
        for k in tensor_dict.keys():
            stylized_tensor_final1 += tensor_dict[k] * intensity_dict[k]

        stylized_image_final1 = tensor_to_image(stylized_tensor_final1)

    st.image(stylized_image_final1, width=512) #use_column_width=True)

    buf = BytesIO()
    stylized_image_final1.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    btn = st.download_button(
          label="Download Image",
          data=byte_im,
          file_name="stylized_image.png",
          mime="image/jpeg",
          )

















