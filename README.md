# 2D Projection clustering

For 3D particle reconstruction, we cluster the particle's 2D projections in different orientations.  
Here we experiment with different image embedding methods using image2vec and subsequently cluster the embeddings using different clustering algorithms.

## Installation
pip install -r requirements.txt --upgrade

## Instructions

1. Constructing 2D projections' dataset from list of pdb identifiers:
```
python src/get_projection_data/build_dataset.py
```
If adding noise to the images is desired:
```
python src/add_noise.py
```
Save configuration for dataset in the function get_config in src/util/input_functions.py

2. Constructing all by all similarity graph:
```
python src/SLICEM/slicem.py
```
3. Constructing top n neighbors or k nearest neighbors graph:
```
python src/SLICEM/slicem_gui.py
```
In the GUI that opens up:
i. In Inputs tab, specify all by all graph as input and click on 'Load inputs'. 

ii.In Network plot tab, specify top n neighbors or k nearest neighbors, and click on 'plot network'

iii.In Outputs tab, specify directory to save files in and click Write edge list. This edge list is the graph used for clustering in the next steps.

For more detailed instructions, refer src/SLICEM/manual.pdf

4. Clustering graph using one or more of different graph clustering methods - kclique, label propagation, walk trap, edge betweenness, conected components, greedy modularity:
```
python src/graph_clustering_igraph.py
python src/graph_clustering_nx.py
```
Evaluate the obtained clusters with:
```
python src/evaluate_graph_clustering.py
```
5. Train siamese neural network on images:
```
python src/make_and_train_siamese_triplets.py
```
Add config to find the trained siamese model during extraction of image embeddings, in the function siamese_embedding in src/siamese_embedding.py and get_image_embedding in src/cluster_image_embeddings.py

6. Constructing image embeddings (using one or more of siamese, efficientnet-b1, efficientnet-b7,resnet-18,vgg, densenet, alexnet) for 2D projections and clustering with best of Birch, OPTICS, Affinity Propagation and DBSCAN. With example arguments:
```
python src/cluster_image_embeddings.py --graph_names "" --graph_types "" --datasets synthetic_noisy --out_dir_suffixes "" --graph_embedding_methods "" --node_attribute_methods '' --embedding_methods densenet siamese vgg alexnet siamese_more_projs_all efficientnet_b1 efficientnet_b7 --eval_SLICEM False --main_results_dir ../results --find_best_clustering_method 1
```
Add config to use these image embeddings as attributes in graph clustering later by updating in the function main of construct_node_embeddings_slicem.py

7. Constructing node embeddings from similarity graph using one or more embedding methods - node2vec, metapath2vec, Watch your step and Graphwave, and constructing graph node embeddings with image embeddings as node attributes using one or more embedding methods from graphSage, attri2vec, gcn, cluster_gcn, gat and APPNP:
```
python src/construct_node_embeddings_slicem.py
```
Add config to find node embeddings in the function slicem_graph_embeddings in read_node_embeddings.py for the clustering step

8. Cluster node embeddings constructed with image embeddings as attributes, with best of Birch, OPTICS, Affinity Propagation and DBSCAN. With example arguments:
```
python cluster_image_embeddings.py --graph_names slicem_edge_list_l2 --graph_types directed --datasets synthetic_noisy --out_dir_suffixes _node_embedding --graph_embedding_methods "" --node_attribute_methods siamese_noisy --embedding_methods attri2vec gcn cluster_gcn gat APPNP graphSage --eval_SLICEM 1 --main_results_dir ../results --find_best_clustering_method 1
```
9. Combine node embeddings with image embeddings and cluster with best of Birch, OPTICS, Affinity Propagation and DBSCAN. With example arguments:
```
python cluster_image_embeddings.py --graph_names slicem_edge_list_l2 --graph_types directed --datasets synthetic_noisy --out_dir_suffixes _combined_externally _combined_internally --graph_embedding_methods metapath2vec wys graphWave node2vec --node_attribute_methods '' --embedding_methods siamese_noisy --eval_SLICEM 1 --main_results_dir ../results --find_best_clustering_method 1
```

### Configuring scripts help
The above instructions use default arguments provided in the script. 
To provide custom arguments for each of the scripts, input arguments options can be viewed by running: 
python script_name.py --help 
For each command, add the desired argument directly on the terminal, 
ex for variable called arg_name with value arg_value: python script_name.py --arg_name arg_value.

## Unit tests
```
python src/test_cluster_image_embeddings.py
```

## References
Code to compute evaluation metrics is adapted from:
https://github.com/marcottelab/super.complex  
https://github.com/marcottelab/protein_complex_maps  

Synthetic and real data, and slicem code is from:
https://doi.org/10.1016/j.jsb.2019.107416
https://github.com/marcottelab/SLICEM

