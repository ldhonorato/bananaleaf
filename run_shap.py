import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, VGG16, InceptionV3, Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import torch
import shap

# Number of ImageNet classes: 1000
# torch.Size([2, 224, 224, 3])
# torch.Size([2, 1000])
# Classes: [132 814]: ['American_egret' 'speedboat']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
experiments_config = {

    "VGG16": {
        "model": VGG16,
        "target": (224, 224),
        "input_shape": (224, 224, 3),
        "batch_size": 16
    },    
    "MobileNetV2": {
        "model": MobileNetV2,
        "target": (224, 224),
        "input_shape": (224, 224, 3),
        "batch_size": 16
    },
    "ResNet50V2": {
        "model": ResNet50V2,
        "target": (224, 224),
        "input_shape": (224, 224, 3),
        "batch_size": 16
    },
}


def predict(img: np.ndarray) -> torch.Tensor:
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, experiments_config[modelname]["target"], cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis=0)
    img = img/255.
    if len(img.shape) > 4:
        img = img[0]

    output_vector = []
    output = MODEL.predict(img)

    output_vector = torch.Tensor(output)

    return output_vector

# def predict(img: np.ndarray) -> torch.Tensor:
    
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img, experiments_config[modelname]["target"], cv2.INTER_CUBIC)
#     img = np.expand_dims(img, axis=0)
#     img = img/255.
#     if len(img.shape) > 4:
#         img = img[0]

#     output_vector = []
#     output = MODEL.predict(img)
#     for _out in output:
#         healthy_prob = np.abs(1-_out[0])
#         segatoka_prob = _out[0]
#         output_vector.append([healthy_prob, segatoka_prob])
#     output_vector = torch.Tensor(output_vector)

#     return output_vector

class_names = ["healthy", "segatoka"]
path_images = "dataset/Banana Leaf Disease Images/processed/test/segatoka"

IMAGES = os.listdir(path_images)

for modelname in experiments_config.keys():
    path_save = os.path.join("results", modelname)
    os.makedirs(path_save, exist_ok=True)

    path_save_checkpoints = os.path.join("models_checkpoints/softmax", modelname)
    filepath = os.path.join(path_save_checkpoints, modelname + "_best.h5")

    MODEL = load_model(filepath)


    for image in IMAGES:
        
        path_image = os.path.join(path_images, image)

        img = cv2.imread(path_image, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, experiments_config[modelname]["target"], cv2.INTER_CUBIC)

        out = predict(img)

        topk = 2
        batch_size = 50
        n_evals = 10000

        # # define a masker that is used to mask out partitions of the input image.
        masker_blur = shap.maskers.Image("blur(128,128)", img.shape)

        # create an explainer with model and image masker
        explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

        # feed only one image
        # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
        shap_values = explainer(np.expand_dims(img, axis=0), max_evals=n_evals, batch_size=batch_size,
                                outputs=shap.Explanation.argsort.flip[:topk])
        s_data, s_values = (shap_values.data, shap_values.values)

        # (torch.Size([1, 224, 224, 3]), (1, 224, 224, 3, 4))
        v_i = s_values[0, :, :, :, 0].copy()
        # print(v_i.min(), v_i.max())
        # vi = (v_i - v_i.min())/(v_i.max() - v_i.min())
        vi = np.where(v_i < 0, 0, v_i)
        vi = vi/vi.max()

        # v_ii = s_values[0, :, :, :, 1].copy()
        # vii = (v_ii - v_ii.min())/(v_ii.max() - v_ii.min())

        path_save_shap = os.path.join(path_save, f"shap-{image}")
        cv2.imwrite(path_save_shap, vi*255)
