import streamlit as st
from PIL import Image
import numpy as np
import torch;
import os;
from torch import nn
from transformers import SegformerFeatureExtractor
os.environ['KMP_DUPLICATE_LIB_OK']='True'


palette = [[0, 0, 0],
           [128, 64, 128],
           [130, 76, 0],
           [0, 102, 0],
           [112, 103, 87],
           [28, 42, 168],
           [48, 41, 30],
           [0, 50, 89],
           [107, 142, 35],
           [70, 70, 70],
           [102, 102, 156],
           [254, 228, 12],
           [254, 148, 12],
           [190, 153, 153],
           [153, 153, 153],
           [255, 22, 96],
           [102, 51, 0],
           [9, 143, 150],
           [119, 11, 32],
           [51, 51, 0],
           [190, 250, 190],
           [112, 150, 146],
           [2, 135, 115],
           [255, 0, 0]]
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)  # remove background class
dir_name = os.path.abspath(os.path.dirname(__file__))
# join the bobrza1.csv to directory to get file path
location = os.path.join(dir_name, 'model.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
model = torch.load(location, map_location=device)
model.eval()


@torch.no_grad()
def predict(img_path):
    image = Image.open(img_path)

    encoding = feature_extractor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.size[::-1],
                                                 mode='bilinear',
                                                 align_corners=False)
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

    np_palette = np.array(palette)
    for label, color in enumerate(np_palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg[..., ::-1]

    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img;



st.title("Image Segmentation with SegFormer")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img=predict(uploaded_file)
    st.image(img)


