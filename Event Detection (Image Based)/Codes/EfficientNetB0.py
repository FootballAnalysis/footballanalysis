
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam,RMSprop, SGD , Nadam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from keras.utils.generic_utils import get_custom_objects
import efficientnet.keras as enet

from keras.backend import sigmoid
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import argparse
import random
import pickle
import json
import os


#################### argument


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-ss", "--save_directory",required=True,
	help="path to save directory")
ap.add_argument("-e", "--epoch", type=int,required=True,
	help="epoch number")
ap.add_argument("-i", "--dims", type=int,required=True,
	help="batch size") 
ap.add_argument("-b", "--batchsize", type=int,required=True,
	help="batch size") 
ap.add_argument("-lr", "--lr", type=float,required=True,
	help="gpu device")   
ap.add_argument("-n", "--num_train", type=int,required=True,
	help="batch size")     
ap.add_argument("-nm", "--num_validation", type=int,required=True,
	help="batch size")  
ap.add_argument("-s", "--neuron", type=int,required=True,
	help="gpu device")  
ap.add_argument("-g", "--gpu",required=True,
	help="gpu device")      
args = vars(ap.parse_args())


############# set parameter

EPOCHS = int(args["epoch"])
INIT_LR = float(args["lr"])
BS = int(args["batchsize"])
inputShape = (int(args["dims"]), int(args["dims"]), 3)
NUM_TRAIN = int(args["num_train"])
NUM_VALIDATION = int(args["num_validation"])
now = datetime.datetime.now()
time_string = str(now.year) + str(now.month) + str(now.day) + "__" + now.strftime("%H:%M")
save_directory= args["save_directory"] + "__" +  time_string + "_d_" + str(args["dims"]) + "_b_" +  str(args["batchsize"]) + "_lr_" + str(args["lr"]) +  "_number_ " + str(args["num_train"])
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
    print("Directory " , save_directory ,  " Created ")
    

#################### GPU


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])


import tensorflow as tf

conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=conf)

#################### data ########################

#################### data generator

train_datagen = ImageDataGenerator(
     rescale=1.0 / 255,
 #   featurewise_center=True,
 #   featurewise_std_normalization=True,
 #   zca_whitening=True,
 #   rotation_range=90,
 #   width_shift_range=0.2,
 #   height_shift_range=0.2,
 #   shear_range=0.2,
 #   zoom_range=0.2,
 #   channel_shift_range=90,
 #   horizontal_flip=True,
 #   vertical_flip=True,
 #   fill_mode="nearest",
 #   brightness_range=[0.0,0.9],
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_dir=args["dataset"] + "/train"
validation_dir = args["dataset"] + "/test"
test_dir = args["dataset"] + "/test"


#################### data generator

print("[INFO] loading images...")
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to target height and width.
    target_size=(int(args["dims"]), int(args["dims"])),
    color_mode="rgb",
    batch_size=BS,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(int(args["dims"]), int(args["dims"])),
    color_mode="rgb",
    batch_size=BS,
    shuffle=False,
    class_mode="categorical",
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(int(args["dims"]), int(args["dims"])),
    color_mode="rgb",
    batch_size=BS,
    shuffle=False,
    class_mode="categorical",
)


############ save labels

print(train_generator.class_indices)
labels = (train_generator.class_indices)
lablelist = []
for key, value in labels.items():
    lablelist.append(key)
arrayLabels = np.array(lablelist)
# binarize both sets of labels
print("[INFO] binarizing labels...")
categoryLB = LabelBinarizer()
#colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(arrayLabels)
# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open("{}/category_lb.pickle".format(save_directory), "wb")
f.write(pickle.dumps(categoryLB))
f.close()


############################ define model

chanDim = -1
# construct both the "category" and "color" sub-networks
class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish_act': SwishActivation(swish_act)})


inputs = Input(shape=inputShape)

model = enet.EfficientNetB0(include_top=False,input_tensor=inputs, pooling='avg', weights='imagenet')
x = model.output

x = BatchNormalization(axis=chanDim)(x)
x = Dropout(0.5)(x)
x = Dense(256, kernel_regularizer='l2')(x)
x = BatchNormalization(axis=chanDim)(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)
x = Dense(256, kernel_regularizer='l2')(x)
x = BatchNormalization(axis=chanDim)(x)
x = Activation(swish_act)(x)
x = Dense(arrayLabels.shape[0])(x)
x = Activation("softmax", name="category_output")(x)
model = Model(
	inputs=inputs,
	outputs=[x],
	name="carnet")
 
 
################# compile and fit model


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["acc"],
)

print(model.summary())

batchX, batchy = train_generator.next()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=NUM_TRAIN // BS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=NUM_VALIDATION // BS,
    verbose=1,
)

################################## save model

# save the model to disk
print("[INFO] serializing network...")
#tf.keras.experimental.export_saved_model(model, args["model"])
model.save("{}/model_saved".format(save_directory))
# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open('{}/file.json'.format(save_directory), 'w'))




scores = model.evaluate_generator(test_generator,NUM_VALIDATION) 
print("Accuracy = ", scores[1])


##################################  detail

f = open("{}/Report_Performance.txt".format(save_directory), "w")
f.write("Time = "+ str(datetime.datetime.now()))
f.write("\n")
f.write("Classes = "+ str(labels))
f.write("\n")
f.write("Epoch = "+ str(EPOCHS))
f.write("\n")
f.write("Init LR = "+ str(INIT_LR))
f.write("\n")
f.write("Batch Size = "+ str(BS))
f.write("\n")
f.write("Input shape = "+ str(inputShape))
f.write("\n")
f.write("Num Train = "+ str(NUM_TRAIN))
f.write("\n")
f.write("Num Vaidation = "+ str(NUM_VALIDATION))
f.write("\n")
f.write("Accuracy = "+ str(scores[1]))
f.write("\n")
f.write("\n")
f.write("\n")
f.write(str(model.summary()))
f.close()


######################################  Confution Matrix and Classification Report

Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = lablelist
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

#plot classification_report
plt.figure(figsize=(20,20))
clf_report = classification_report(test_generator.classes,
                                   y_pred,
                                   target_names=target_names,
                                   output_dict=True)
                                   
ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap="binary" )
ax.set_title('Classification Report - Category');
plt.savefig('{}/Classification_Report_Category.png'.format(save_directory), dpi=200, format='png', bbox_inches='tight')   



#plot confusion matrix - category
plt.figure(figsize=(20,20))
ax = sns.heatmap(confusion_matrix(test_generator.classes,y_pred), cmap="binary",annot=True,fmt="d")
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix - Category'); 
ax.set_xticklabels(target_names, rotation=90)
ax.set_yticklabels(target_names, rotation=0) # reversed order for y
plt.savefig("{}/Confusion_Matrix_Category.jpg".format(save_directory))





##############################  plot accuracy  and loss

# plot the total loss, category loss, and color loss
lossNames = ["loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), history.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the losses figure
plt.tight_layout()
plt.savefig("{}/losses.png".format(save_directory))
plt.close()


# create a new figure for the accuracies
accuracyNames = ["acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), history.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the accuracies figure
plt.tight_layout()
plt.savefig("{}/accs.png".format(save_directory))
plt.close()






