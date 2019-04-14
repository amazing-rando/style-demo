from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import numpy as np


def preprocess_image(image_path, h, w):
    '''
    Takes input of image location as well as desired height and width and
    returns the loaded, resized image file after preprocessing it for the
    VGG19 convolutional neural network.
    '''
    image = load_img(image_path, target_size=(h, w))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = vgg19.preprocess_input(image)
    return image

def postprocess_image(image):
    '''
    Post-processes an image to make it properly viewable.
    '''
    #Remove zero-center by mean pixel.
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68

    #Convert from BGR to RGB color space.
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype("uint8")
    return image

def content_loss(target, combined):
    '''
    Content loss function.
    Ensures that top layer of neural net will have a similar view of the target
    and generated images.
    '''
    loss = K.sum(K.square(combined - target))
    return loss

def gram_matrix(image):
    '''
    Computes Gram matrix of input image.
    This maps the correlations found in the orignal feature matrix.
    '''
    features = K.batch_flatten(K.permute_dimensions(image, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combined, h, w):
    '''
    Style loss function.
    '''
    style_g = gram_matrix(style)
    combined_g = gram_matrix(combined)
    channels = 3
    size = h * w
    loss = K.sum(K.square(style_g - combined_g) /
            (4 * (channels ** 2) * (size ** 2)))
    return loss

def total_variation_loss(image, h, w):
    '''
    Total variation loss function.
    Encourages spatial continuity in generated image.
    '''
    a = K.square(image[:, :h - 1, :w - 1, :] - image[:, 1:, :w - 1, :])
    b = K.square(image[:, :h - 1, :w - 1, :] - image[:, :h - 1, 1:, :])
    loss = K.sum(K.pow(a + b, 1.25))
    return loss

def general_loss(model, weights, combined, h, w):
    #Dict mapping layer names to activation tensors
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    #Define content and style layers.
    content_layer = "block5_conv2"
    style_layers = ["block1_conv1",
                    "block2_conv1",
                    "block3_conv1",
                    "block4_conv1",
                    "block5_conv1"]

    #Initialize general loss variable.
    loss = K.variable(0)

    #Compute content loss and add to general loss.
    layer_f = outputs_dict[content_layer]
    target_f = layer_f[0, :, :, :]
    combined_f = layer_f[2, :, :, :]
    loss = loss + (weights[0] * content_loss(target_f, combined_f))

    #For each of the style layers
    for layer_name in style_layers:
        #Compute style loss and add to general loss.
        layer_f = outputs_dict[layer_name]
        style_f = layer_f[1, :, :, :]
        combined_f = layer_f[2, :, :, :]
        sl = style_loss(style_f, combined_f, h, w)
        loss = loss + ((weights[1] / len(style_layers)) * sl)
    
    #Compute total variation loss and add to general loss.
    loss = loss + (weights[2] * total_variation_loss(combined, h, w))
    return loss

class evaluator(object):
    '''
    SciPy's L-BFGS algorithm can only be applied to flat vectors and loss
    function must be passed separate from gradient values.
    
    This evaluator allows for the algorithm to be applied to a 3D image array
    while passing loss and gradients at the same time.

    This greatly speeds things up.
    '''
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, image, get_loss_and_grads, h, w):
        assert self.loss_value is None
        image = image.reshape((1, h, w, 3))
        outs = get_loss_and_grads([image])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, image, get_loss_and_grads, h, w):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
