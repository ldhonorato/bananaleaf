import tensorflow as tf
import numpy as np
import cv2
import shap
import torch

CLASS_NAMES = ['SAUDÁVEL', 'DOENTE']
MODEL = tf.keras.models.load_model('VGG16_best.h5')

def predict(img: np.ndarray) -> tf.Tensor:
    
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

def predict_class(image, model):

    image_tf = tf.cast(image, tf.float32)
    #image_int = tf.cast(image, tf.int8)
    image_tf = tf.image.resize(image_tf, [224, 224])

    image2 = np.expand_dims(image_tf, axis = 0)

    prediction = model.predict(image2)
    
    return prediction

def generate_shap(image, preds):
    image_tf = tf.cast(image, tf.float32)
    #image_int = tf.cast(image, tf.int8)
    image_tf = tf.image.resize(image_tf, [224, 224])

    image2 = np.expand_dims(image_tf, axis = 0)
    vi = None
    if np.argmax(preds) > 0:
        topk = 2
        batch_size = 50
        n_evals = 1000

        # # define a masker that is used to mask out partitions of the input image.
        masker_blur = shap.maskers.Image("blur(128,128)", image_tf.shape)

        # create an explainer with model and image masker
        explainer = shap.Explainer(predict, masker_blur, output_names=CLASS_NAMES)

        # feed only one image
        # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
        shap_values = explainer(image2, max_evals=n_evals, batch_size=batch_size,
                                outputs=shap.Explanation.argsort.flip[:topk])
        s_data, s_values = (shap_values.data, shap_values.values)

        # (torch.Size([1, 224, 224, 3]), (1, 224, 224, 3, 4))
        v_i = s_values[0, :, :, :, 0].copy()
        # print(v_i.min(), v_i.max())
        # vi = (v_i - v_i.min())/(v_i.max() - v_i.min())
        vi = np.where(v_i < 0, 0, v_i)
        vi = vi/vi.max()
        vi = vi*255

        heatmap = cv2.applyColorMap(vi.astype(np.uint8), cv2.COLORMAP_JET)
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        vi = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        # vi = vi*255
    return vi

test_image = cv2.imread('20210219_112815.jpg', 1)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (224,224), cv2.INTER_CUBIC)
# test_image = Image.open(file)

preds = predict_class(np.asarray(test_image), MODEL)

index_pred = np.argmax(preds)

result = CLASS_NAMES[np.argmax(preds)]

output = 'Esta planta está ' + result
print(output)

if index_pred > 0:
    print('Executando explicação do diagnóstico (pode demorar cerca de 1 minuto)....')
    shap = generate_shap(np.asarray(test_image), preds)
    cv2.imwrite('teste.png', shap)

print('Pronto')