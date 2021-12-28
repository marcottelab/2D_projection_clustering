import os
import numpy as np
from tensorflow.keras import metrics
from cluster_image_embeddings import get_config, read_data
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import resnet

from loguru import logger


dataset = 'synthetic'

images_file_name,images_true_labels,sep,index_start,out_dir_orig = get_config(dataset)

out_dir_orig = out_dir_orig + '_evaluate_siamese_model'
if not os.path.exists('./' + out_dir_orig):
        os.mkdir('./' + out_dir_orig)

# Setting logger
logger.add('./' + out_dir_orig + '/log_file.txt',level="INFO")

data, gt_lines,gt_names = read_data(images_file_name, images_true_labels, sep)

converted_data =[tf.keras.preprocessing.image.img_to_array(Image.fromarray(np.uint8(data_arr*255)).convert('RGB')) for data_arr in data]

embedding = tf.keras.models.load_model('siamese_embedding_model.tf')
logger.info('Loaded model')

#train_dataset = tf.data.experimental.load('train_dataset.tf', element_spec=None)
#logger.info('Loaded train dataset')

#sample = next(iter(train_dataset))
#anchor_images = [converted_data[0],converted_data[2], converted_data[4], converted_data[6]]
#positive_images = [converted_data[1],converted_data[3], converted_data[5], converted_data[7]]
#negative_images = [converted_data[49],converted_data[50], converted_data[51], converted_data[52]]
anchor_images = [converted_data[0],converted_data[2], converted_data[4], converted_data[6],converted_data[1]]
positive_images = [converted_data[1],converted_data[3], converted_data[5], converted_data[7], converted_data[5]]
negative_images = [converted_data[49],converted_data[50], converted_data[51], converted_data[52], converted_data[53]]

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
logger.info('Constructing positive dataset...')
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
logger.info('Constructing negative dataset...')
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

logger.info('Putting together full dataset...')

train_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
#dataset = dataset.shuffle(buffer_size=1024)



# Batching is needed
train_dataset = train_dataset.batch(32, drop_remainder=False)
# Below two lines not necessary but can keep for speed
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)

sample = next(iter(train_dataset)) # The samplegives the next batch, not one image combination
#print(type(sample)) # tuple

anchor, positive, negative = sample

anchor_images_test=tf.convert_to_tensor([converted_data[0]])
print(tf.shape(anchor_images_test))
#positive_images = tf.data.Dataset.from_tensors(converted_data[1])
#negative_images =  tf.data.Dataset.from_tensors(converted_data[100])

print(type(anchor)) # eager tensor
print(tf.shape(anchor))
#print(anchor)
#tf.Tensor([  4 350 350   3], shape=(4,), dtype=int32)
anchor = tf.convert_to_tensor([converted_data[0],converted_data[2], converted_data[4], converted_data[6],converted_data[1]])
positive = tf.convert_to_tensor([converted_data[1],converted_data[3], converted_data[5], converted_data[7], converted_data[5]])
negative = tf.convert_to_tensor([converted_data[49],converted_data[50], converted_data[51], converted_data[52], converted_data[53]])

print(type(anchor)) # eager tensor
print(tf.shape(anchor))
preprocessed_anchor = resnet.preprocess_input(anchor)
preprocessed_positive = resnet.preprocess_input(positive)
preprocessed_negative = resnet.preprocess_input(negative)
#anchor_embedding, positive_embedding, negative_embedding = ( embedding(resnet.preprocess_input(anchor)), embedding(resnet.preprocess_input(positive)), embedding(resnet.preprocess_input(negative)))

anchor_embedding, positive_embedding, negative_embedding = ( embedding(preprocessed_anchor), embedding(preprocessed_positive), embedding(preprocessed_negative))

print(type(anchor_embedding)) # eager tensor
print(tf.shape(anchor_embedding))
cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
logger.info("Positive similarity: {}", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
logger.info("Negative similarity {} ", negative_similarity.numpy())

