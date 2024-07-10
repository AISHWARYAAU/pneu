import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)


def loading_model():
  fp = "cnnModel.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()

st.write("""
# Pneumonia Detector
by Ahmedh Shamsudeen and Araiz Asad :sunglasses:
""")

filler_image = Image.open('AIface.jpeg')
st.image(filler_image)
st.write("Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.\n\n\nSource: https://www.webmd.com/lung/understanding-pneumonia-basics") 

temp = st.file_uploader("Upload A Chest X-Ray Image")
st.write("Link to chest X-ray Dataset for testing purposes if needed https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")

buffer = temp
temp_file = NamedTemporaryFile(delete=False)


if buffer:
  temp_file.write(buffer.getvalue())
  st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:
  loaded_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')
  # Preprocessing the image
  pp_loaded_img = image.img_to_array(loaded_img)
  pp_loaded_img = pp_loaded_img/255
  pp_loaded_img = np.expand_dims(pp_loaded_img, axis=0)
  #predict
  image_preds= cnn.predict(pp_loaded_img)
  print(image_preds)
  if image_preds>= 0.5:
      out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(image_preds[0][0]))

  else: 
      out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-image_preds[0][0]))


  st.success(out)

  image = Image.open(temp)
  st.image(image,use_column_width=True)
