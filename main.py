import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import style
import numpy as np

st.title("Style Transfer")

img = st.sidebar.file_uploader("Upload Custom Image", type=["jpg", "JPEG", "png"])

style_name = st.sidebar.selectbox(
    'Select Style',
    ('mosaic','autumn','udnie','wave')
)

def get_image(img_path):
  img = load_img(img_path)
  img = img_to_array(img, dtype=np.float32)
  return img
def post_process(img):
  img = tensorflow.clip_by_value(img, 0, 255)
  img = img.numpy()
  img = tensorflow.squeeze(img)
  img = img.numpy()
  img = img.astype(int)
  return img

#main
if img is not None:
  input_image = img
  st.write("### Source image:")
  image = Image.open(input_image)
  st.image(image,width=500)
  clicked = st.button("Stylize")
  if clicked:
    if style_name != "autumn":
      model = tensorflow.keras.models.load_model("style/"+style_name+"/"+style_name+"/saved_models")
    elif style_name == "autumn":
      model = tensorflow.keras.models.load_model("style/"+"rain_princess"+"/"+"rain_princess"+"/saved_models")
    img = get_image(img)
    img_tensor = tensorflow.convert_to_tensor(img)
    img_tensor = tensorflow.expand_dims(img_tensor, 0)
    output = model(img_tensor)
    st.write("### Output image:")
    output_array = output.numpy()
    output_img = Image.fromarray(np.uint8(output_array[0]))
    st.image(output_img, width=350)
else:
  st.subheader('Please upload an image!')
