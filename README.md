# 2D Projection clustering

For 3D particle reconstruction, we cluster the particle's 2D projections in different orientations.  
Here we experiment with different image embedding methods using image2vec and subsequently cluster the embeddings using different clustering algorithms.

## Installation 
Requires:  
scikit-learn  
mrcfile  
numpy  
pandas  
pytorch  
img2vec_pytorch  
Refer https://github.com/christiansafka/img2vec to install image2vec.  
Make sure all the models of interest (ex: 'alexnet', 'vgg','densenet','resnet-18') are present in the img_to_vec.py file.  

## Instructions

Data used in our test experiments is from this work: https://doi.org/10.1016/j.jsb.2019.107416.  
Pls contact meghana.palukuri@utexas.edu for this data.  

Make sure the data is present in the specified format and in a folder called data at the same level as the file cluster_image_embeddings.py
```
python cluster_image_embeddings.py
```

## Unit tests
```
python test_cluster_image_embeddings.py
```

## References
Code to compute evaluation metrics is adapted from:  
https://github.com/marcottelab/super.complex  
https://github.com/marcottelab/protein_complex_maps  

