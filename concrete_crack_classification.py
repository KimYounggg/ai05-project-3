# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:31:02 2022

@author: Kim Young
"""

import tensorflow as tf
import matplotlib.pyplot as plt

#Load images and split into train-validation sets
filepath = r"C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Datasets\Concrete Crack Images"
IMG_SIZE = (180, 180)

train_dataset = tf.keras.utils.image_dataset_from_directory(filepath,
                                                             validation_split = 0.2,
                                                             shuffle = True,
                                                             subset = 'training',
                                                             seed = 12345,
                                                             image_size = IMG_SIZE,
                                                             batch_size = 32)

val_dataset = tf.keras.utils.image_dataset_from_directory(filepath,
                                                            validation_split = 0.2,
                                                            shuffle = True,
                                                            subset = 'validation',
                                                            seed = 12345,
                                                            image_size = IMG_SIZE,
                                                            batch_size = 32)

#%%
#Further split into validation-test sets using validation set
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 5)
validation_dataset = val_dataset.skip(val_batches // 5)

#%%
class_names = train_dataset.class_names

plt.figure(figsize = (10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        
#%%
#Create prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size = AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size = AUTOTUNE)

#%%
#Create data augmentation model
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))

#%%
#Display image augmentation examples
for images,labels in train_dataset_pf.take(1):
    first_image = images[0]
    print(labels)
    plt.figure(figsize=(10,10))
    
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
        
#%%
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
IMG_SHAPE = IMG_SIZE + (3,)

#Create the base model
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

#%%
#Freeze the base model
base_model.trainable = False
base_model.summary()

#%%
#Add classification layer using global average pooling
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#Add output layer
prediction_layer = tf.keras.layers.Dense(1)

#%%
#Create model using functional API
inputs = tf.keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs,outputs)
model.summary()

#%%
#Compile model
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)

model.compile(optimizer = adam, loss = loss, metrics = ['accuracy'])

#%%
EPOCHS = 10
log_path = r'C:\\Users\\Kim Young\\Desktop\\SHRDC\\Deep Learning\\TensorFlow Deep Learning\\Tensorboard\\logs'
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 2)

history = model.fit(train_dataset_pf, validation_data = validation_dataset_pf, epochs = EPOCHS, callbacks = [tb_callback, es_callback])

#%%
#Perform fine tuning
#Unfreeze base_model
#base_model.trainable = True

# Freeze first n number of layers
# fine_tune_at = 100 

# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
    
# rmsprop = tf.keras.optimizers.RMSprop(learning_rate = 0.00001)        
# model.compile(optimizer = rmsprop, loss = loss, metrics = ['accuracy'])
# model.summary()

#%%
#fine_tune_epoch = 10
#total_epoch = EPOCHS + fine_tune_epoch

#history_fine = model.fit(train_dataset_pf,
#                         validation_data = validation_dataset_pf,
#                         epochs=total_epoch,initial_epoch = history.epoch[-1],
#                         callbacks = [tb_callback])

#%%
#Evaluate model
test_loss, test_accuracy = model.evaluate(test_dataset_pf)

print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#Make predictions
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

#Apply sigmoid to output
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5,0,1)

print(f'Prediction: {predictions.numpy()}')
print(f'Labels: {label_batch}')

#%%
#Display the predictions
plt.figure(figsize = (10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    prediction = class_names[predictions[i]]
    label = class_names[label_batch[i]]
    plt.title(f"Prediction: {prediction}, Actual: {label}")
    plt.axis('off')
