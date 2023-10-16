# Web App for Arbitrary Neural Style Transfer
Streamlit Web App for Arbitrary Neural Style Transfer with Multiple Styles.

![alt text](https://raw.githubusercontent.com/dorukcanga/Style-Transfer-Streamlit-App/main/tofi_vangogh.jpg?raw=true)

## App Link

[![Streamlit APP](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multiple-style-transfer.streamlit.app)

## Features

This app allows users to apply multiple styles to their photos or to create GIFs with style transitions.
There are several style options available in the app. Users may select one or more style images from existing options or they may upload style images via upload option.

Weights of multiple styles are configurable. Users may adjust style intensity levels to create images with more or less intense styles.

Also, GIFs with style transtion can be created. Users may create GIFs of original content images with transition between 2 styles.

## References

This app uses fast arbitrary style transfer model from TensorFlow Hub. Details can be found [here](https://www.tensorflow.org/tutorials/generative/style_transfer).

Model used in the app can be found [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

[Streamlit Documentation](https://docs.streamlit.io)

