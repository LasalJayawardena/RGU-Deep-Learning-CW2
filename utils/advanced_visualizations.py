
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot
from matplotlib.pyplot import imshow
%matplotlib inline


def visualize_umap(array_images_set, labels, no_of_images, width_of_image, height_of_image):
    
    # Reshaping the data and converting them to RGB images
    array_images_set = array_images_set.reshape(-1, 28,28)
    array_images_set = np.stack((array_images_set,)*3, axis=-1)
    
    images_set = []
    if (no_of_images == 100):
        # For 100 items, 10 from each class
        no_of_images = 100
        count = 0
        for label in range(10):  # because the dataset has labels from 0 to 9
            count+=10
            for i in range(len(array_images_set)):
                if labels[i] == label:
                    images_set.append(array_images_set[i])
                    if len(images_set) >= count:
                        break
    else:
        images_set = array_images_set


    feature_and_image_data = []
    for i in range(no_of_images):
        pil_image = Image.fromarray(images_set[i]).resize((28,28))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50,50))
        image = image.flatten()
        feature_and_image_data.append([image, pil_image])
        
        
    image_features, pil_images  = zip(*feature_and_image_data)
    
    # Performing PCA
    image_features = np.array(image_features)
    pca_model = PCA(n_components=100) # was 300, changed to 100 because i only had 102 images when testing
    pca_model.fit(image_features)
    pca_features = pca_model.transform(image_features)
    
    
    # Performing UMAP
    umap_reducer = umap.UMAP(
            n_neighbors = 7,
            min_dist = 1
    )
    
    pca_features_array = np.array(pca_features)
    pipeline = Pipeline([('scaling', StandardScaler()), ('umap', umap_reducer)])
    image_embedding = pipeline.fit_transform(pca_features_array)
    
    x, y = image_embedding[:,0], image_embedding[:,1]
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))

    
    # Plotting
    image_width = width_of_image
    image_height = height_of_image
    maximum_dimension = 100
    full_plot_image = Image.new('RGBA', (image_width, image_height))
    
    for image, x, y in zip(pil_images, x, y):
        img_tile = image
        rs = max(1, img_tile.width/maximum_dimension, img_tile.height/maximum_dimension)
        img_tile = img_tile.resize((int(img_tile.width/rs), int(img_tile.height/rs)), Image.ANTIALIAS)
        full_plot_image.paste(img_tile, (int((image_width-maximum_dimension)*x), int((image_height-maximum_dimension)*y)), mask=img_tile.convert('RGBA'))
        
    matplotlib.pyplot.figure(figsize = (16,12))
    imshow(full_plot_image)



def visualize_tsne(array_images_set, labels, no_of_images, width_of_image, height_of_image):
    
    # Reshaping the data and converting them to RGB images
    array_images_set = array_images_set.reshape(-1, 28,28)
    array_images_set = np.stack((array_images_set,)*3, axis=-1)
    
    images_set = []
    if (no_of_images == 100):
        # For 100 items, 10 from each class
        no_of_images = 100
        count = 0
        for label in range(10):
            count+=10
            for i in range(len(array_images_set)):
                if labels[i] == label:
                    images_set.append(array_images_set[i])
                    if len(images_set) >= count:
                        break
    else:
        images_set = array_images_set


    feature_and_image_data = []
    for i in range(no_of_images):
        pil_image = Image.fromarray(images_set[i]).resize((28,28))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (50,50))
        image = image.flatten()
        feature_and_image_data.append([image, pil_image])
        
        
    image_features, pil_images  = zip(*feature_and_image_data)
    
    # Performing PCA
    image_features = np.array(image_features)
    pca_model = PCA(n_components=100) # was 300, changed to 100 because i only had 102 images when testing
    pca_model.fit(image_features)
    pca_features = pca_model.transform(image_features)
    
    # Performing TSNE
    pca_image_features = np.array(pca_features)
    tsne_model = TSNE(n_components=2, learning_rate=350, perplexity=30, angle=0.2, verbose=2).fit_transform(pca_image_features)
    
    x, y = tsne_model[:,0], tsne_model[:,1]
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))
    
    # Plotting
    image_width = width_of_image
    image_height = height_of_image
    max_dim = 100
    full_image = Image.new('RGBA', (image_width, image_height))
    
    for image, x, y in zip(pil_images, x, y):
        img_tile = image
        rs = max(1, img_tile.width/max_dim, img_tile.height/max_dim)
        img_tile = img_tile.resize((int(img_tile.width/rs), int(img_tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(img_tile, (int((image_width-max_dim)*x), int((image_height-max_dim)*y)), mask=img_tile.convert('RGBA'))
        
    matplotlib.pyplot.figure(figsize = (16,12))
    imshow(full_image