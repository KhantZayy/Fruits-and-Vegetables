import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_Fruits and Vegetables_CNN_model.h5")
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    return predicted_index

#sidebar

st.sidebar.title(':red[Dashboard]')
app_mode = st.sidebar.selectbox('You can choose here.',['Home', 'About'],                
            placeholder="Select Pages...")

#Main page

if app_mode == 'Home':
    st.title(':blue[FRUITS & VEGETABLES RECOGNITION SYSTEM]')
    image = 'wallhaven-o5921p.jpg'
    st.image(image)
    st.subheader('Fruits and Vegetables Prediction', divider='red')
    test_img = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_img,width=4,use_column_width=True)
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_img)
        with open("class_set.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success(f'This is {label[result_index]}')

                   
elif app_mode == 'About':
    st.title(':red[About our app]')
    st.write('Our application’s primary objective is to discern the type of fruit or vegetable depicted within an image.')
    st.subheader('Context', divider= 'red')
    st.text("This dataset encompasses images of various fruits and vegetables, \nproviding a diverse collection for image recognition tasks.\nThe included food items are:")
    st.markdown(':blue[***Fruits🍇***]')
    st.code('Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango')
    st.markdown(':green[***Vegetables🥬***]')
    st.code('Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant')

    st.header('About our future', divider='red')
    st.write('We will endeavor to incorporate an increased variety of fruits and vegetables into our classification system to categorize the diverse types of these produce items.')
