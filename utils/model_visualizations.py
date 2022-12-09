import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import cv2

def visualize_layer_feature_maps_for_image(model, layer_name, image, image_title, is_transfer_learning=True, num_filters=None, rows=8, columns=8, figsize=(12, 12)):
    """
    Visualize Feature maps of a particular model layer
    """

    if is_transfer_learning:
        inputs = model.layers[1].inputs
        outputs = [model.layers[1].get_layer(layer_name).output]
    else: 
        inputs = model.inputs
        outputs = [model.get_layer(layer_name).output]
        
    feature_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    img = image[np.newaxis, ...]
    feature_output = feature_model.predict(img)

    fig = plt.figure(figsize=figsize)
    
    if num_filters is None:
        num_filters = columns * rows
        
    for i in range(1, num_filters +1):
        ax = plt.subplot(rows, columns, i)
        ax.set_xticks([])  
        ax.set_yticks([])
        plt.imshow(feature_output[0, :, :, i-1], cmap='gray') 

    fig.suptitle(f"Visualization of Feature Maps in Layer: {layer_name.title()} for an image of a {image_title}", size=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    


def visualize_layer_filters(model, layer_name, is_transfer_learning=True, num_filters=None, rows=8, columns=8, figsize=(12, 12)):
    """
    Visualize Kernel filters of a particular model layer
    """

    if is_transfer_learning:
        viz_layer = model.layers[1].get_layer(layer_name)
    else: 
        viz_layer = model.get_layer(layer_name)

    filters, biases = viz_layer.get_weights()

    fig = plt.figure(figsize=figsize)
    
    if num_filters is None:
        num_filters = columns * rows

    for i in range(1, num_filters +1):
        n_filter = filters[:, :, :, i-1]
        ax = plt.subplot(rows, columns, i)
        ax.set_xticks([])  
        ax.set_yticks([])
        plt.imshow(n_filter[:, :, 0], cmap='gray') 

    fig.suptitle(f"Visualization of Filters in Layer: {layer_name.title()}", size=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    


class Grad_CAM:
    """
    Implementation of Grad-CAM: Gradient-weighted Class Activation Mapping for any Model
    """
    def __init__(self, model, layer_name, is_transfer_learning=True):
        if is_transfer_learning:
            self.model = model.layers[1] 
        else:
            self.model = model
        self.layer_name = layer_name
        
        
    def compute_heatmap(self, image, eps=1e-8):

        gradient_model = tf.keras.models.Model(inputs=[self.model.inputs],
                  outputs=[self.model.get_layer(self.layer_name).output,
                  self.model.output])  
        
                
        with tf.GradientTape() as tape:
            # obtain the loss of the image
            img = image[np.newaxis, ...]
            inputs = tf.cast(img, tf.float32)
            (convolution_outputs, loss) = gradient_model(inputs)

        # compute the gradients using auto differentitation
        gradients = tape.gradient(loss, convolution_outputs)
        
        # use guided backprop to compute the guided gradients
        casted_convolution_outputs = tf.cast(convolution_outputs > 0, "float32")
        casted_gradients = tf.cast(gradients > 0, "float32")
        guided_gradients = casted_convolution_outputs * casted_gradients * gradients
 
        # discard the batch dimension
        convolution_outputs = convolution_outputs[0]
        guided_gradients = guided_gradients[0]
        
        # respect to the weights, compute the ponderation of the filters 
        weights = tf.reduce_mean(guided_gradients, axis=(0, 1))
        class_activation_map = tf.reduce_sum(tf.multiply(weights, convolution_outputs), axis=-1)
        
        # resize spatial dimenion to input images ize
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(class_activation_map.numpy(), (w, h))

        # normalize the heatmap 
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):

        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        overlayed_image = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, overlayed_image)
    
    def plot_grad_cam(self, image, figsize=(8, 8), image_title="", image_size=(28,28)):
        
        plt.style.use(['default'])
        heatmap = self.compute_heatmap(image)
        heatmap = cv2.resize(heatmap, image_size)
        (heatmap, overlayed_img) = self.overlay_heatmap(heatmap, image)
        fig = plt.figure(figsize=figsize)
    
        ax = plt.subplot(1, 3, 1)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Original Image", y=0, pad=-25, verticalalignment="top")
        plt.imshow(image)
        
        ax = plt.subplot(1, 3, 2)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Gradient Map", y=0, pad=-25, verticalalignment="top")
        plt.imshow(heatmap)
        
        ax = plt.subplot(1, 3, 3)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Image with GradCAM", y=0, pad=-25, verticalalignment="top")
        plt.imshow(overlayed_img)
        

        fig.suptitle(f"Gradient-weighted Class Activation Mapping (Grad-CAM) Visualization of a {image_title} in the {self.layer_name.title()} layer", size=18)
        fig.tight_layout(rect=[0, -0.2, 1, 1.7])
        plt.show()


class Saliency_Map:
    """
    Implementation of Saliency Maps for any Model
    """
    def __init__(self, model):
        self.model = model
        
        
    def compute_map(self, image, eps=1e-18):

        img = image[np.newaxis, ...]
        image = tf.Variable(img, dtype=float)
                    
        with tf.GradientTape() as tape:
            # obtain the loss of the image            
            prediction = self.model(image, training=False)
            
            class_indexes_sorted = np.argsort(prediction.numpy().flatten())[::-1]
            loss = prediction[0][class_indexes_sorted[0]]

        # compute the gradients using auto differentitation
        gradients = tape.gradient(loss, image)
        
        gradients_abs = tf.math.abs(gradients)
        gradients_max = np.max(gradients_abs, axis=3)[0]
        
        ## normalize to range between 0 and 1
        arr_min, arr_max  = np.min(gradients_max), np.max(gradients_max)
        saliency_map = (gradients_max - arr_min) / (arr_max - arr_min + eps)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        return saliency_map
    
    def overlay_map(self, s_map, image, alpha=0.6,
        colormap=cv2.COLORMAP_TURBO):

        # overlay the heatmap on the input image
        s_map = cv2.applyColorMap(s_map, colormap)
        overlayed_image = cv2.addWeighted(image, alpha, s_map, 1 - alpha, 0)

        return (s_map, overlayed_image)
    
    def plot_saliency_map(self, image, figsize=(8, 8), image_title="", image_size=(28,28)):
        
        plt.style.use(['default'])
        s_map = self.compute_map(image)
        s_map = cv2.resize(s_map, image_size)
        (s_map, overlayed_img) = self.overlay_map(s_map, image)
        fig = plt.figure(figsize=figsize)
    
        ax = plt.subplot(1, 3, 1)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Original Image", y=0, pad=-25, verticalalignment="top")
        plt.imshow(image)
        
        ax = plt.subplot(1, 3, 2)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Saliency Map", y=0, pad=-25, verticalalignment="top")
        plt.imshow(s_map)
        
        ax = plt.subplot(1, 3, 3)
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_title("Image with Saliency Map", y=0, pad=-25, verticalalignment="top")
        plt.imshow(overlayed_img)
        

        fig.suptitle(f"Saliency Map Visualization of a {image_title}", size=18)
        fig.tight_layout(rect=[0, -0.2, 1, 1.7])
        plt.show()


