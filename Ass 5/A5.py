from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib import cm


n_classes = 6

# =============================================================================
# def getVGGFeatures(img, layerName):
#     base_model = VGG16(weights='imagenet')
#     model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
#     img = img.resize((224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
# 
#     internalFeatures = model.predict(x)
# 
#     return internalFeatures
# =============================================================================
def getVGGFeatures(fileList, layerName):
    #Initial Model Setup
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
    
    #Confirm number of files passed is what was expected
    rArray = []
    print ("Number of Files Passed:")
    print(len(fileList))

    for iPath in fileList:
        try:
            #Read Image
            img = image.load_img(iPath)
            #Update user as to which image is being processed
            print("Getting Features " +iPath)
            #Get image ready for VGG16
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            #Generate Features
            internalFeatures = model.predict(x)
            rArray.append(internalFeatures.flatten())            
        except:
            print ("Failed "+ iPath)
    return rArray

def cropImage(image, x1, y1, x2, y2):
    return image.crop((x1, y1, x2, y2))

def standardizeImage(image, x, y):
    return image.resize((x, y))

def preProcessImages(images):
    directory = "Processed"
    datafile = "downloadedFiles.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for line in open(datafile):
        temp_string = line.split()[5]
        temp_list = temp_string.split(',')
        out_list = []
        for i in temp_list:
            out_list.append(int(i))
        filename = line.split()[7]
        try:
            im = Image.open(images + "/" + filename) 
            im = cropImage(im, out_list[0], out_list[1], out_list[2], out_list[3])
            im = standardizeImage(im, 60, 60)
            im.save(directory + "/" + filename)
        except(OSError):
            print("Fail to open the image " + filename)
        
        
def visualizeWeight():
    #You can change these parameters if you need to
    utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
    X_train, X_val, y_train, y_val = train_test_split(preProcessedImages, labels, test_size=0.2, random_state=1)
    
    train_num = X_train.shape[0]
    val_num = X_val.shape[0]
    X_train = X_train.reshape(train_num, 3600)
    X_val = X_val.reshape(val_num, 3600)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    
    X_train /= 255
    X_val /= 255
    
    Y_train = keras.utils.to_categorical(y_train, n_classes)
    Y_val = keras.utils.to_categorical(y_val, n_classes)
    
    model = Sequential()
    model.add(Dense(784, input_shape=(3600,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(256, activation = 'relu')) 
    #model.add(Dropout(0.2))  
    #model.add(Dense(128, activation = 'relu')) 
    #model.add(Dropout(0.2))                           
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    history = model.fit(X_train, Y_train,
          batch_size=128, epochs=50,
          verbose=2,
          validation_data=(X_val, Y_val))
    
    plt.figure(figsize=(20,20))
    plt.plot(history.history['loss'], label = 'training data')
    plt.plot(history.history['val_loss'], label = 'validation data')
    plt.title('Data Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
    return model

def trainFaceClassifier_VGG(extractedFeatures, labels):
    X_train, X_val, y_train, y_val = train_test_split(extractedFeatures, labels, test_size=0.2, random_state=1)
    
    Y_train = keras.utils.to_categorical(y_train, n_classes)
    Y_val = keras.utils.to_categorical(y_val, n_classes)
    
    model = Sequential()
    model.add(Dense(512, input_shape=(100352,))) #100352 for block4, 25988 for block5
    model.add(Activation('relu'))                            
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    history = model.fit(X_train, Y_train,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(X_val, Y_val))
    
    plt.figure(figsize=(20,20))
    plt.plot(history.history['loss'], label = 'training data')
    plt.plot(history.history['val_loss'], label = 'validation data')
    plt.title('Data Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
    return model

if __name__ == '__main__':
    print("Your Program Here")
    #preProcessImages("uncropped")
# =============================================================================
#     im = Image.open("test.jpg") 
#     im_1 = cropImage(im, 2381,736,3268,1623)
#     im_1.show()
#     im_2 = standardizeImage(im_1, 60, 60)
#     im_2.show()
# =============================================================================
    all_images = []
    all_images_path = []
    all_labels = []
    for filename in os.listdir('Processed'):
        try:
            im = Image.open("Processed/" + filename).convert('L')
            all_images.append(np.array(im))
            all_images_path.append("Processed/" + filename)
            new_filename = ''
            for i in filename:
                if i.isdigit():
                    break
                else:
                    new_filename += i
            all_labels.append(new_filename)
        except(OSError):
            print("Fail to open the image " + filename)
    
    X = np.empty(shape = (len(all_images), 60, 60))
    y = []
    for i in range(X.shape[0]):
        X[i] = all_images[i]
    for i in range(len(all_labels)):
        if all_labels[i] == 'gilpin':
            y.append(1)
        if all_labels[i] == 'harmon':
            y.append(2)
        if all_labels[i] == 'butler':
            y.append(3)
        if all_labels[i] == 'radcliffe':
            y.append(4)
        if all_labels[i] == 'vartan':
            y.append(5)
        if all_labels[i] == 'bracco':
            y.append(0)
        
#     Part 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    face_classifer = trainFaceClassifier(X_train, y_train)
    
    test_num = X_test.shape[0]
    X_test = X_test.reshape(test_num, 3600)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = keras.utils.to_categorical(y_test, n_classes)
    
    score = face_classifer.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    names = [weight.name for layer in face_classifer.layers for weight in layer.weights]
    weights = face_classifer.get_weights()
    
    for name, weight in zip(names, weights):
        print(name, weight.shape)
    
    actor1 = np.empty(shape = (784, ))
    actor2 = np.empty(shape = (784, ))
    j = 0
    for i in weights[2]:
        actor1[j] = i[0]
        actor2[j] = i[1]
        j+=1
    
    fig = plt.figure(1, figsize=(20, 20)) 
    ax = fig.gca()  
    heatmap = ax.imshow(actor1.reshape((28,28)), cmap = cm.coolwarm)  
    fig.colorbar(heatmap, shrink = 0.5, aspect=5) 
    fig.show()
    
    fig2 = plt.figure(2, figsize=(20, 20)) 
    ax2 = fig2.gca()  
    heatmap2 = ax2.imshow(actor2.reshape((28,28)), cmap = cm.coolwarm)  
    fig2.colorbar(heatmap2, shrink = 0.5, aspect=5) 
    fig2.show()

#     Part 3
    VGG_list = getVGGFeatures(all_images_path, "block4_pool")
    VGG_array = np.asarray(VGG_list)
    X_train_VGG, X_test_VGG, y_train_VGG, y_test_VGG = train_test_split(VGG_array, y, test_size=0.2, random_state=1)
    face_classifer_VGG = trainFaceClassifier_VGG(X_train_VGG, y_train_VGG)
    Y_test_VGG = keras.utils.to_categorical(y_test_VGG, n_classes)
    score_VGG = face_classifer_VGG.evaluate(X_test_VGG, Y_test_VGG, verbose=0)
    print('Test loss:', score_VGG[0])
    print('Test accuracy:', score_VGG[1])
