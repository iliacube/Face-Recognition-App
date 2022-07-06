# Face verification app

# Goal statement: Build an app that will be able to verify the user's face by distinguishing it from other faces.

# Methodology followed: 
#    Build a dataset by using local camera and taking snapshots. We need positive pairs of pictures (where both pictures belong to the same face) and negative pairs (for this, we have used the Labelled Faces in the Wild dataset). 

# For this reason we create three directories, the first is the Anchor (first images in the pairs), the second is the Positive (second images in the pairs that match the anchors) and the third is the Negative directory (third images in the pairs that differentiate from the anchors).





# Import dependencies:

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf 
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid



# Create directories to save the samples:

# Setup Paths
POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data', 'anchor')
VER_PATH = os.path.join('application_data', 'verification_images')
INP_PATH = os.path.join('application_data', 'input_image')

os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)
os.makedirs(VER_PATH)
os.makedirs(INP_PATH)


# Uncompress the Labelled Faces in the Wild dataset and move them to /data/negative path:

get_ipython().system('tar -xf lfw.tgz')


# Move LFW images to data/negative

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)




# Use OpenCV to establish connection to the webcam and take pictures. The frame must be cropped to be 250x250 end keyboard actions must be specified in order to save the pictures in the corresponding folders and break gracefully:

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize frame to 250x250 px
    frame = frame[190:190+250, 230:230+250, :]
    
    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)     

    # Collect verifications
    if cv2.waitKey(1) & 0XFF == ord('v'):
        imgname = os.path.join(VER_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


# Release the webcam        
cap.release()
# Close the image show frame
cv2.destroyAllWindows()


# Data augmentation process

def data_augmentation(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100),np.random.randint(100)))
        
        data.append(img)
        
    return data


# Apply the data augmentation function to the whole Anchor and Positive directories:

def apply_data_augmentation(file_path):
    for file_name in os.listdir(os.path.join(file_path)):
        img_path = os.path.join(file_path, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_augmentation(img)

        for image in augmented_images:
            cv2.imwrite(os.path.join(file_path, '{}.jpg'.format(uuid.uuid1())), image.numpy())



apply_data_augmentation(ANC_PATH)
apply_data_augmentation(POS_PATH)



# Take 3000 sample file paths in a random shuffled manner from each directory:

anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)



# Define function to load images from filepaths, decode jpegs, scale pixel values to facilitate training and resize images to 105x105 (input dimension for our conv layer):

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img / 255.0
    return img



# Create labeled dataset with pairs of positives, negatives and labels (1,0)

positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)



# Define function to preprocess the sample pairs:

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)



# Build dataloader pipeline to apply the preprocessing process, cache the dataset and shuffle in order to have both positive and negative pairs in training and testing sets:

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)



# Split dataset to training and testing at 0.7, create batches of 16 and prefetch to improve latency and throughput:

train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data)*0.7))
test_data = test_data.take(round(len(data)*0.3))
test_data = test_data.batch(16, drop_remainder =True)
test_data = test_data.prefetch(8)



# Build the embedding layer:

def make_embedding():
    inp = Input(shape=(105,105,3), name='input_image')
    
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()



# Defining class of custom layer for L1 Distance between the embeddings:

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)



# Build the final siamese network model so as to process the pairs of images, calculate the L1 distance and do the binary classification:

def make_siamese_model():
    
    # Handle inputs
    input_image = Input(name='input_img', shape=(105,105,3))
    validation_image = Input(name='validaption_img', shape=(105,105,3))
    
    
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siamese_network')


siamese_model = make_siamese_model()



# Define Loss and optimizer:

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(0.0001)



# Create checkpoints for the training:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)



# Define the training step process, by using the GradientTape.gradient and the apply_gradients(). Convert it to a graph representation for better performance:


@tf.function
def train_step(batch):
    
    # Record all operations
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
            
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss



# Training process to loop through the batches, perform the training step and accumulate statistics for the precision and recall for each epoch:


def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2], verbose=0)
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
            
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        




EPOCHS = 20

train(train_data, EPOCHS)




# Evaluate the model on the testing set:

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val], verbose=0)
    r.update_state(y_true, yhat)
    p.update_state(y_true, yhat)

print(r.result().numpy(), p.result().numpy())


# Visualize the data

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.imshow(test_input[9])
plt.subplot(1,2,2)
plt.imshow(test_val[9])
plt.show()
print(y_true[9])



# Save the model:


siamese_model.save('siamesemodel_v1.h5')


model = tf.keras.models.load_model('siamesemodel_v1.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})



# Verification function - Compare the input image with each one of the verification images. If the number of images that have been predicted as positive (p>detection_threshold) is greater than the verification_threshold, then the person is verified:


def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
       
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    
    return results, verified


# Use the webcam to capture a photo as an input for the verification:


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[190:190+250, 230:230+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
    
        results, verified = verify(model, 0.5, 0.5)
        print(verified)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

