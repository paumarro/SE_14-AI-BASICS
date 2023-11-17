import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#Loading the dataset from Keras
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#Helper function to show some (12) sample images
def show_images(train_images, class_names, train_labels, nb_samples = 12, nb_row = 4):
	plt.figure(figsize=(12, 12))
	for i in range(nb_samples):
		plt.subplot(nb_row, nb_row, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[train_labels[i][0]])
	plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           	'dog', 'frog', 'horse', 'ship', 'truck']

show_images(train_images, class_names, train_labels)

#Normalizing the pixl values (0-1) to ensure scale invariance and fast convergence during the training
max_pixel_value = 255

train_images = train_images / max_pixel_value 
test_images = test_images / max_pixel_value

#(One-hot encoding) Convert the categorical labels to a numerical format so that they can be easily processed by the neural net 
train_labels = tf.keras.utils.to_categorical(train_labels, len(class_names))
test_labels =  tf.keras.utils.to_categorical(test_labels, len(class_names))


#Implementing the architecture of the network (Sequential() to define the model, add() to add a layer)

# Variables
INPUT_SHAPE = (32, 32, 3)
FILTER1_SIZE = 32 #Figure out what exactly the size represents
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3) #Play around with the filter shape
POOL_SHAPE = (4, 4) #Experiment: Play around with the size of the pooling layer & remove it from a convlayer
FULLY_CONNECT_NUM = 128
NUM_CLASSES = len(class_names)

# Model architecture implementation
model = Sequential()
model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Flatten())
model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Model training
BATCH_SIZE = 32 # indicating that each training iteration processes 32 samples before updating the model's parameters.
EPOCHS = 30 # Number of times the model is trained on the entire training dataset.

METRICS = metrics=['accuracy',
               	Precision(name='precision'),
               	Recall(name='recall')]

model.compile(optimizer='adam',
          	loss='categorical_crossentropy',
          	metrics = METRICS)

# Train the model
training_history = model.fit(train_images, train_labels,
                	epochs=EPOCHS, batch_size=BATCH_SIZE,
                	validation_data=(test_images, test_labels))

# Evaluate the model
def show_performance_curve(training_result, metric, metric_label):
    
	train_perf = training_result.history[str(metric)]
	validation_perf = training_result.history['val_'+str(metric)]
	intersection_idx = np.argwhere(np.isclose(train_perf,
                                            	validation_perf, atol=1e-2)).flatten()[0]
	intersection_value = train_perf[intersection_idx]
    
	plt.plot(train_perf, label=metric_label)
	plt.plot(validation_perf, label = 'val_'+str(metric))
	plt.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')
    
	plt.annotate(f'Optimal Value: {intersection_value:.4f}',
         	xy=(intersection_idx, intersection_value),
         	xycoords='data',
         	fontsize=10,
         	color='green')
            	 
	plt.xlabel('Epoch')
	plt.ylabel(metric_label)
	plt.legend(loc='lower right')

show_performance_curve(training_history, 'accuracy', 'accuracy')

show_performance_curve(training_history, 'precision', 'precision')


test_predictions = model.predict(test_images)

test_predicted_labels = np.argmax(test_predictions, axis=1)

test_true_labels = np.argmax(test_labels, axis=1)

cm = confusion_matrix(test_true_labels, test_predicted_labels)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm)

cmd.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.show()