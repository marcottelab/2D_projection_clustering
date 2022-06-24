# 2D Projection clustering

For 3D particle reconstruction, we cluster the particle's 2D projections in different orientations.  
Here we experiment with different image embedding methods using image2vec and subsequently cluster the embeddings using different clustering algorithms.

## Installation
pip install -r requirements.txt --upgrade

## Instructions

Data used in our test experiments is from this work: https://doi.org/10.1016/j.jsb.2019.107416.  
Pls contact meghana.palukuri@utexas.edu for this data.  

Make sure the data is present in the specified format and in a folder called data at one directory level above the file cluster_image_embeddings.py
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

