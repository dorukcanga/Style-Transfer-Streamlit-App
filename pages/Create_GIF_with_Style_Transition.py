import numpy as np
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub
import PIL.Image
import cv2
from io import BytesIO
import base64

import streamlit as st

from utils import utils, load_hub_model, load_image, tensor_to_image, preprocess_image

st.set_page_config(layout="wide")

st.sidebar.title('Style Transfer Application')

logo_url =  utils.logo_url
st.sidebar.image(logo_url)

st.title('Configurable Style Transfer Application with Multiple Styles')
st.header('Create GIF with Style Transition')


####################################################################################################
# Load Model from Tensorflow Hub
####################################################################################################

model = load_hub_model(utils.model_url)

####################################################################################################
# Content & Style Image Upload Sections
####################################################################################################

col1, col2, col3 = st.columns(3)

with col1:

    #Upload Content Image
    st.subheader('Upload Content Photo')
    
    tab1 = st.tabs(["Upload Content Image"])
    
    content_image = st.file_uploader("Please Upload JPG File", key=0)
    if content_image is not None:
        content_image2 = preprocess_image(content_image)

    
with col2:
    
    #Upload or Select Style Image
    st.subheader('Upload or Choose Style Photo - 1')
    
    tab1, tab2 = st.tabs(["Choose Style", "Upload New Style"])
    
    #Select Style Image Option
    with tab1: 
        selected_style1 = st.selectbox(
            'Choose From Existing Styles:',
             list(utils.default_style_dict.keys()), key=1, index=None)
        
        if selected_style1 is not None:
            selected_style1_2 = load_image(utils.default_style_dict[selected_style1], size=512)
    
    #Upload Style Image Option
    with tab2:
        selected_style1 = st.file_uploader("Please Upload JPG File", accept_multiple_files=False, key=2)
        if selected_style1 is not None:
            selected_style1_2 = preprocess_image(selected_style1)
            
with col3:
    
    #Upload or Select Style Image
    st.subheader('Upload or Choose Style Photo - 2')
    
    tab1, tab2 = st.tabs(["Choose Style", "Upload New Style"])
    
    with tab1: 
        selected_style2 = st.selectbox(
            'Choose From Existing Styles:',
             list(utils.default_style_dict.keys()), key=3, index=None)
        
        if selected_style2 is not None:
            selected_style2_2 = load_image(utils.default_style_dict[selected_style2], size=512)
    
    #Upload Style Image Option
    with tab2:
        selected_style2 = st.file_uploader("Please Upload JPG File", accept_multiple_files=False, key=4)
        if selected_style2 is not None:
            selected_style2_2 = preprocess_image(selected_style2)
    

####################################################################################################
# Content & Style Image Printing Sections
####################################################################################################

col1, col2, col3= st.columns(3)

with col1:
    st.subheader('Content Photo:')
    try:
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(content_image, caption='Content Image', use_column_width=True)
    except Exception:
        st.text("Content image\nwill be shown here\nonce uploaded.")

with col2:
    st.subheader('Style Photo - 1:')
    try:
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(selected_style1_2, caption='Style Image - 1', use_column_width=True)
    except Exception:
        st.text("Second style image\nwill be shown here\nonce uploaded or choosen.")

with col3:
    st.subheader('Style Photo - 2:')
    try:
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(selected_style2_2, caption='Style Image - 2', use_column_width=True)
    except Exception:
        st.text("Second style image\nwill be shown here\nonce uploaded or choosen.")
            
            
####################################################################################################
# Set original style ratio 
####################################################################################################

st.subheader('Set Original Content Intensity Ratio in the Final GIF')

if content_image == None:
    st.warning('Please upload content image  to set intensity.')
else:
    content_intensity = st.number_input('Content Intensity Level (%):', min_value=0, max_value=100, value=0, step=1)
    content_intensity = content_intensity / 100
    
    
####################################################################################################
# Transfer Style(s)
####################################################################################################    

st.subheader('Create GIF with Style Transition')

exe = st.button("Run Model")

if exe:
    with st.spinner('Running...'):
        
        stylized_tensor1 = model(tf.constant(content_image2), tf.constant(selected_style1_2))[0]
        stylized_image1 = tensor_to_image(stylized_tensor1)

        stylized_tensor2 = model(tf.constant(content_image2), tf.constant(selected_style2_2))[0]
        stylized_image2 = tensor_to_image(stylized_tensor2)
        #Style Transfer
            
        #Create GIF
        gif_images = []
        num_image = 20
        remain_intensity = 1 - content_intensity
        for i in range(num_image+1):

            style_intensity1 = remain_intensity * (num_image-i)/num_image
            style_intensity2 = remain_intensity * i/num_image

            stylized_image_final = style_intensity1 * stylized_tensor1 + style_intensity2 * stylized_tensor2 + content_image2 * content_intensity
            temp_img = tensor_to_image(stylized_image_final)
            gif_images.append(temp_img)
            
            
    gif_images[0].save('export.gif', format = 'GIF', save_all = True, loop = 0, append_images = gif_images)
    
    
    file_ = open('export.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    

    with open('export.gif', 'rb') as file:
        btn = st.download_button(
            label='Download GIF',
            data=file,
            file_name='stylized_gif.gif',
            mime='image/gif'
          )
        
