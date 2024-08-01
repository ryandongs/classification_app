import streamlit as st
from PIL import Image
from clf import predict

#st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Classification")
st.write("")

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels, score = predict(file_up)

    # print out the top 5 prediction labels with scores
    st.write("Prediction (index, name)", labels, ",   Score: ", score)
