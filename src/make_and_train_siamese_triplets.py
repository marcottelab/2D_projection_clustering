# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:59:42 2021

@author: Meghana
"""
from cluster_image_embeddings import get_config, read_data
from itertools import combinations
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from loguru import logger
import pickle
import faulthandler 

faulthandler.enable()
def get_image_triples(dataset_name, target_shape):
    images_file_name,images_true_labels,sep,index_start,out_dir_orig, sep2 = get_config(dataset_name)

        
    data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep, sep2)

    if len(target_shape) == 0:
        target_shape = data[0].shape



    logger.info('Lenth of data {}',len(data))
    logger.info('Lenth of complexes {}',len(gt_lines))
    logger.info('Lenth of complex names {}',len(gt_names))
    siamese_pairs = []
    siamese_triples_with_10_sampled_negs = [] # Format: anchor, positive, negative

    # Starting from index 0
    all_images = set(range(len(data)))

    for cluster_set in gt_lines:
        cluster_pairs = list(combinations(cluster_set, 2))
        siamese_pairs = siamese_pairs + cluster_pairs
        negatives = all_images - cluster_set
        
        for pair in cluster_pairs:
            sampled_negatives = []
            for i in range(n_negs_sampled):
                sampled_negatives.append(random.choice(list(negatives)))
                
            siamese_triples_with_10_sampled_negs = siamese_triples_with_10_sampled_negs + [(pair[0],pair[1],str(neg)) for neg in sampled_negatives]
           

    logger.info(siamese_triples_with_10_sampled_negs[0])
    logger.info('N triples {}',len(siamese_triples_with_10_sampled_negs))
    logger.info('Converting images...')
    if data[0].shape is not target_shape:
        # Resize
        converted_data =[tf.keras.preprocessing.image.img_to_array(Image.fromarray(np.uint8(data_arr*255)).convert('RGB').resize(target_shape)) for data_arr in data] 
    else:
        converted_data =[tf.keras.preprocessing.image.img_to_array(Image.fromarray(np.uint8(data_arr*255)).convert('RGB')) for data_arr in data] 

    logger.info('Making anchor images list...')
    #anchor_images = [data[int(triple[0])] for triple in siamese_triples_with_10_sampled_negs]
    anchor_images = [converted_data[int(triple[0])-index_start] for triple in siamese_triples_with_10_sampled_negs]

    logger.info('Making positive images list...')        
    #positive_images = [data[int(triple[1])] for triple in siamese_triples_with_10_sampled_negs]
    positive_images = [converted_data[int(triple[1])-index_start] for triple in siamese_triples_with_10_sampled_negs]

    logger.info('Making negative images list...')
    #negative_images = [data[int(triple[2])] for triple in siamese_triples_with_10_sampled_negs]
    negative_images = [converted_data[int(triple[2])-index_start] for triple in siamese_triples_with_10_sampled_negs]

    return anchor_images, positive_images, negative_images

def get_tf_dataset(positive_images, negative_images, anchor_images):
    logger.info('N pos {}',len(positive_images))
    logger.info('N neg = {}', len(negative_images))
    logger.info('N anchor = {}', len(anchor_images))


    logger.info('Constructing anchor dataset...')
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    logger.info('Constructing positive dataset...')
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)  
    logger.info('Constructing negative dataset...')  
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)            
            
    logger.info('Putting together full dataset...')  

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    return dataset
#dataset_name = 'synthetic_more_projs'
#dataset_name = 'synthetic'
#dataset_name = 'synthetic_noisy'
#dataset_name = 'synthetic_more_projs_noisy'
#datasets = ['real']
datasets = ['real', 'synthetic_more_projs_noisy']

exp_name = '_'.join(datasets)
out_dir_orig = exp_name + 'maketriplets'
if not os.path.exists('./' + out_dir_orig):
    os.mkdir('./' + out_dir_orig)
    
# Setting logger
logger.add('./' + out_dir_orig + '/log_file.txt',level="INFO")
n_negs_sampled = 10
target_shape = (100,100)
anchor_images = []
positive_images = []
negative_images = []

for dataset_name in datasets:
    
    anchor_imgs, positive_imgs, negative_imgs = get_image_triples(dataset_name, target_shape)
    anchor_images = anchor_images + anchor_imgs
    positive_images = positive_images + positive_imgs
    negative_images = negative_images + negative_imgs


dataset = get_tf_dataset(positive_images, negative_images, anchor_images)

logger.info('Splitting dataset...')  

# Let's now split our dataset in train and validation.
image_count = len(anchor_images)
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

batch_size = 2 # Originally 32
logger.info('Training data samples {}',len(train_dataset))
logger.info('Validation data samples = {}', len(val_dataset))
train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)

val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(AUTOTUNE)

logger.info('Writing datasets to file...')
tf.data.experimental.save(train_dataset,exp_name+'train_dataset.tf')
tf.data.experimental.save(val_dataset,exp_name+'val_dataset.tf')


"""
## Setting up the embedding generator model
Our Siamese Network will generate embeddings for each of the images of the
triplet. To do this, we will use a ResNet50 model pretrained on ImageNet and
connect a few `Dense` layers to it so we can learn to separate these
embeddings.
We will freeze the weights of all the layers of the model up until the layer `conv5_block1_out`.
This is important to avoid affecting the weights that the model has already learned.
We are going to leave the bottom few layers trainable, so that we can fine-tune their weights
during training.
"""

logger.info('Setting base resnet...')
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

logger.info('Configuring other layers...')

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

"""
## Setting up the Siamese Network model
The Siamese network will receive each of the triplet images as an input,
generate the embeddings, and output the distance between the anchor and the
positive embedding, as well as the distance between the anchor and the negative
embedding.
To compute the distance, we can use a custom layer `DistanceLayer` that
returns both values as a tuple.
"""


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

"""
## Putting everything together
We now need to implement a model with custom training loop so we can compute
the triplet loss using the three embeddings produced by the Siamese network.
Let's create a `Mean` metric instance to track the loss of the training process.
"""


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.
    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


"""
## Training
We are now ready to train our model.
"""

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))

logger.info('Training model...')
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

"""
## Inspecting what the network has learned
At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.
We can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the
similarity between embeddings.
Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""
sample = next(iter(train_dataset))

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)

try:
    with open(exp_name+'siamese_embedding_model.pkl','wb') as f:
        pickle.dump(embedding,f)
except:
    embedding.save(exp_name+'siamese_embedding_model.tf')
    

"""
Finally, we can compute the cosine similarity between the anchor and positive
images and compare it with the similarity between the anchor and the negative
images.
We should expect the similarity between the anchor and positive images to be
larger than the similarity between the anchor and the negative images.
"""

cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
logger.info("Positive similarity: {}", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
logger.info("Negative similarity {} ", negative_similarity.numpy())


"""
## Summary
1. The `tf.data` API enables you to build efficient input pipelines for your model. It is
particularly useful if you have a large dataset. You can learn more about `tf.data`
pipelines in [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data).
2. In this example, we use a pre-trained ResNet50 as part of the subnetwork that generates
the feature embeddings. By using [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en),
we can significantly reduce the training time and size of the dataset.
3. Notice how we are [fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en#fine-tuning)
the weights of the final layers of the ResNet50 network but keeping the rest of the layers untouched.
Using the name assigned to each layer, we can freeze the weights to a certain point and keep the last few layers open.
4. We can create custom layers by creating a class that inherits from `tf.keras.layers.Layer`,
as we did in the `DistanceLayer` class.
5. We used a cosine similarity metric to measure how to 2 output embeddings are similar to each other.
6. You can implement a custom training loop by overriding the `train_step()` method. `train_step()` uses
[`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape),
which records every operation that you perform inside it. In this example, we use it to access the
gradients passed to the optimizer to update the model weights at every step. For more details, check out the
[Intro to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
and [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=en).
"""


    

