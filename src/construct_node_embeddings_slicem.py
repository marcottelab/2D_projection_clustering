# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:29:43 2022

@author: Meghana
"""


import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk, UniformRandomMetaPathWalk
from gensim.models import Word2Vec
from pickle import dump as pkl_dump

from stellargraph.mapper import (
CorruptedGenerator,
FullBatchNodeGenerator,
GraphSAGENodeGenerator,
HinSAGENodeGenerator,
ClusterNodeGenerator,)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
    

def get_graph_embeddings(G, combined, embedding_to_combine,dataset_type, graph_name, graph_type):
    print(G.info())
    
    #node_embedding_method = 'node2vec'
    #node_embedding_method = 'metapath2vec'
    #node_embedding_method = 'wys'
    #node_embedding_method = 'graphWave'
    
    # All the below ones need node features as attributes. Get image embeddings and use them as features
    # node_embedding_method = 'graphSage'
    # node_embedding_method = 'attri2vec'
    # node_embedding_method = 'gcn'
    node_embedding_method = 'graphSage_dgi' # Fix error
    #node_embedding_method = 'cluster_gcn'
    #node_embedding_method = 'gat'
    #node_embedding_method = 'APPNP'
    
    if combined:
        node_embedding_methods = ['graphSage','attri2vec','gcn','cluster_gcn','gat','APPNP']
    else:
        node_embedding_methods = ['node2vec','metapath2vec','wys','graphWave']
    
    
    for node_embedding_method in node_embedding_methods:
        if node_embedding_method == 'node2vec':
            rw = BiasedRandomWalk(G)
            
            walks = rw.run(
                nodes=list(G.nodes()),  # root nodes
                length=100,  # maximum length of a random walk
                n=10,  # number of random walks per root node
                p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
                q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
            )
            
            print("Number of random walks: {}".format(len(walks))) #1000
            str_walks = [[str(n) for n in walk] for walk in walks]
            
            model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)
            
            node_ids = model.wv.index_to_key  # list of node IDs
            node_embeddings = (
                model.wv.vectors
            )  # numpy.ndarray of size number of nodes times embeddings dimensionality    
        elif node_embedding_method == 'metapath2vec':
            walk_length = 100  # maximum length of a random walk to use throughout this notebook
            
            # # specify the metapath schemas as a list of lists of node types.
            # metapaths = [
            #     ["user", "group", "user"],
            #     ["user", "group", "user", "user"],
            #     ["user", "user"],
            # ]
            
            metapaths = [['default','default']]
            
            # Create the random walker
            rw = UniformRandomMetaPathWalk(G)
            walks = rw.run(
                nodes=list(g.nodes()),  # root nodes
                length=walk_length,  # maximum length of a random walk
                n=1,  # number of random walks per root node
                metapaths=metapaths,  # the metapaths
            )        
            # walks = rw.run(
            #     nodes=list(G.nodes()),  # root nodes
            #     length=walk_length,  # maximum length of a random walk
            #     n=1,  # number of random walks per root node  
            # )
            
            print("Number of random walks: {}".format(len(walks))) #1000
            str_walks = [[str(n) for n in walk] for walk in walks]
            
            model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)
            
            node_ids = model.wv.index_to_key  # list of node IDs
            node_embeddings = (
                model.wv.vectors
            )  # numpy.ndarray of size number of nodes times embeddings dimensionality
        elif node_embedding_method == 'wys': # Watch your step
            from stellargraph.core import StellarGraph
            from stellargraph.mapper import AdjacencyPowerGenerator
            from stellargraph.layer import WatchYourStep
            from stellargraph.losses import graph_log_likelihood
            from tensorflow.keras import optimizers, Model, layers, regularizers
            import tensorflow as tf
            from sklearn import preprocessing, feature_extraction, model_selection
            tf.random.set_seed(1234)
            generator = AdjacencyPowerGenerator(G, num_powers=10)
            wys = WatchYourStep(
            generator,
            num_walks=80,
            embedding_dimension=128,
            attention_regularizer=regularizers.l2(0.5),)
            x_in, x_out = wys.in_out_tensors()
            model = Model(inputs=x_in, outputs=x_out)
            model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))
            epochs = 100
            batch_size = 10
            train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)
            history = model.fit(
            train_gen, epochs=epochs, verbose=1, steps_per_epoch=int(len(G.nodes()) // batch_size))
            node_embeddings = wys.embeddings()
            node_ids = list(G.nodes())
        elif node_embedding_method == 'graphWave': # Graph wave
            from stellargraph.mapper import GraphWaveGenerator
            from stellargraph import StellarGraph
            from scipy.sparse.linalg import eigs
            import tensorflow as tf
            from tensorflow.keras import backend as K
            sample_points = np.linspace(0, 100, 50).astype(np.float32)
            degree = 20
            scales = [5, 10]
            
            generator = GraphWaveGenerator(G, scales=scales, degree=degree)
            
            embeddings_dataset = generator.flow(
                node_ids=G.nodes(), sample_points=sample_points, batch_size=1, repeat=False
            )
            
            node_embeddings = np.array([np.squeeze(x.numpy()) for x in embeddings_dataset])
            node_ids = list(G.nodes())
        elif node_embedding_method == 'graphSage': # Unsupervised graphSage
            import networkx as nx
            import pandas as pd
            import numpy as np
            import os
            import random
            
            import stellargraph as sg
            from stellargraph.data import EdgeSplitter
            from stellargraph.mapper import GraphSAGELinkGenerator
            from stellargraph.layer import GraphSAGE, link_classification
            from stellargraph.data import UniformRandomWalk
            from stellargraph.data import UnsupervisedSampler
            from sklearn.model_selection import train_test_split
            
            from tensorflow import keras
            from sklearn import preprocessing, feature_extraction, model_selection
            from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
            from sklearn.metrics import accuracy_score
            
            from stellargraph import globalvar
            
            from stellargraph import datasets
            
            nodes = list(G.nodes())
            number_of_walks = 1
            length = 5
            unsupervised_samples = UnsupervisedSampler(
                G, nodes=nodes, length=length, number_of_walks=number_of_walks
            )
            batch_size = 50
            epochs = 4
            num_samples = [10, 5]
            generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
            train_gen = generator.flow(unsupervised_samples)
            layer_sizes = [50, 50]
            graphsage = GraphSAGE(
                layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
            )
            # Build the model and expose input and output sockets of graphsage, for node pair inputs:
            x_inp, x_out = graphsage.in_out_tensors()
            prediction = link_classification(
                output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
            )(x_out)
            model = keras.Model(inputs=x_inp, outputs=prediction)
            
            model.compile(
                optimizer=keras.optimizers.Adam(lr=1e-3),
                loss=keras.losses.binary_crossentropy,
                metrics=[keras.metrics.binary_accuracy],
            )
            
            history = model.fit(
                train_gen,
                epochs=epochs,
                verbose=1,
                use_multiprocessing=False,
                workers=4,
                shuffle=True,
            )
            
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            from stellargraph.mapper import GraphSAGENodeGenerator
            import pandas as pd
            
            x_inp_src = x_inp[0::2]
            x_out_src = x_out[0]
            embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
            node_ids = list(G.nodes()) # CHECKKKKKKKKKKKKKKK
            node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
            
            node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
        elif node_embedding_method == 'attri2vec': # Consider giving image embeddings as input and combining directly 
            import networkx as nx
            import pandas as pd
            import numpy as np
            import os
            import random
            
            import stellargraph as sg
            from stellargraph.data import UnsupervisedSampler
            from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
            from stellargraph.layer import Attri2Vec, link_classification
            
            from tensorflow import keras
            
            from pandas.core.indexes.base import Index
            
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.metrics import accuracy_score
            
            from stellargraph import datasets
            nodes = list(G.nodes())
            number_of_walks = 4
            length = 5
            
            unsupervised_samples = UnsupervisedSampler(
                G, nodes=nodes, length=length, number_of_walks=number_of_walks
            )
            
            batch_size = 50
            epochs = 4
            
            generator = Attri2VecLinkGenerator(G, batch_size)
            train_gen = generator.flow(unsupervised_samples)
            
            layer_sizes = [128]
            attri2vec = Attri2Vec(
                layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
            )
            
            # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
            x_inp, x_out = attri2vec.in_out_tensors()
            
            prediction = link_classification(
                output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
            )(x_out)
            
            model = keras.Model(inputs=x_inp, outputs=prediction)
            
            model.compile(
                optimizer=keras.optimizers.Adam(lr=1e-3),
                loss=keras.losses.binary_crossentropy,
                metrics=[keras.metrics.binary_accuracy],
            )
            
            history = model.fit(
                train_gen,
                epochs=epochs,
                verbose=2,
                use_multiprocessing=False,
                workers=1,
                shuffle=True,
            )
            
            x_inp_src = x_inp[0]
            x_out_src = x_out[0]
            embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
            
            node_ids=list(G.nodes())
            node_gen = Attri2VecNodeGenerator(G, batch_size).flow(node_ids)
            node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)
        elif node_embedding_method == 'gcn':
            from tensorflow.keras import Model
            import tensorflow as tf

            fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
            gcn_model = GCN(layer_sizes=[128], activations=["relu"], generator=fullbatch_generator)
            
            corrupted_generator = CorruptedGenerator(fullbatch_generator)
            gen = corrupted_generator.flow(G.nodes())
            
            
            infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
            x_in, x_out = infomax.in_out_tensors()
            
            model = Model(inputs=x_in, outputs=x_out)
            model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
            epochs = 100
            es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
            history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
            
            x_emb_in, x_emb_out = gcn_model.in_out_tensors()
            # for full batch models, squeeze out the batch dim (which is 1)
            x_out = tf.squeeze(x_emb_out, axis=0)
            emb_model = Model(inputs=x_emb_in, outputs=x_out)
            node_embeddings = emb_model.predict(fullbatch_generator.flow(G.nodes()))
            node_ids = list(G.nodes())
        elif node_embedding_method == 'cluster_gcn':   
            epochs = 100
            es = EarlyStopping(monitor="loss", min_delta=0, patience=20)    
            fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
            
            cluster_generator = ClusterNodeGenerator(G, clusters=12, q=4)
            cluster_gcn_model = GCN(
                layer_sizes=[128], activations=["relu"], generator=cluster_generator
            )
            node_embeddings, node_ids = run_deep_graph_infomax(es, fullbatch_generator,
                cluster_gcn_model, cluster_generator, epochs=epochs, reorder=cluster_reorder
            )    
        elif node_embedding_method == 'gat':
            epochs = 100
            es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
            fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
            gat_model = GAT(
            layer_sizes=[128], activations=["relu"], generator=fullbatch_generator, attn_heads=8,)
            node_embeddings, node_ids = run_deep_graph_infomax(es, fullbatch_generator,gat_model, fullbatch_generator, epochs=epochs)
        elif node_embedding_method == 'APPNP':
            epochs = 100
            es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
            fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
            appnp_model = APPNP(
            layer_sizes=[128], activations=["relu"], generator=fullbatch_generator)
            node_embeddings, node_ids = run_deep_graph_infomax(es, fullbatch_generator, appnp_model, fullbatch_generator, epochs=epochs)
        elif node_embedding_method == 'graphSage_dgi':
            epochs = 100
            fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
            es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
            graphsage_generator = GraphSAGENodeGenerator(G, batch_size=1000, num_samples=[5])
            graphsage_model = GraphSAGE(
                layer_sizes=[128], activations=["relu"], generator=graphsage_generator)
            node_embeddings, node_ids = run_deep_graph_infomax(es, fullbatch_generator,
                graphsage_model, graphsage_generator, epochs=epochs)
    
        # Write node embeddings to file
        import numpy as np
        with open('../data/' + dataset_type + '_dataset/graph_embeddings/' + graph_name + graph_type + '_stellar_' + node_embedding_method + embedding_to_combine + '.npy', 'wb') as f:
            np.save(f, node_embeddings)
            
        with open('../data/' + dataset_type + '_dataset/graph_embeddings/' + graph_name + graph_type + '_stellar_' + node_embedding_method+ embedding_to_combine + '_node_ids.list', 'wb') as f:
            pkl_dump(node_ids,f)    
            
def run_deep_graph_infomax(es, fullbatch_generator,
    base_model, generator, epochs, reorder=lambda sequence, subjects: subjects
):
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())
    infomax = DeepGraphInfomax(base_model, corrupted_generator)

    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
    history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])

    x_emb_in, x_emb_out = base_model.in_out_tensors()
    # for full batch models, squeeze out the batch dim (which is 1)
    if generator.num_batch_dims() == 2:
        x_emb_out = tf.squeeze(x_emb_out, axis=0)

    emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)

    node_embeddings = emb_model.predict(fullbatch_generator.flow(G.nodes()))
    node_ids = list(G.nodes())
    return node_embeddings, node_ids


def cluster_reorder(sequence, subjects):
    # shuffle the subjects into the same order as the sequence yield
    return subjects[sequence.node_order]

from argparse import ArgumentParser as argparse_ArgumentParser

parser = argparse_ArgumentParser("Input parameters")
parser.add_argument("--dataset_type", default="synthetic_more_projs", help="Dataset name, opts: real, synthetic, synthetic_noisy")
parser.add_argument("--combined_opts", nargs='+', type = bool, default=[True, False], help="Flag to combine image embeddings with graph, opts: True, False or both")
parser.add_argument("--embeddings_to_combine", nargs='+', default=['siamese_more_projs'], help="Image embeddings, specify list")
parser.add_argument("--graph_name_opts", nargs='+', default=["slicem_edge_list"], help="Name of slicem graph")
parser.add_argument("--graph_types", nargs='+', default=["directed","undirected"],help="Type of graph - directed, undirected or both")

args = parser.parse_args()
    

dataset_type = args.dataset_type

#combined_opts = [True,False]
combined_opts = args.combined_opts

#embeddings_to_combine = ['densenet','siamese','vgg','alexnet','siamese_more_projs_all','efficientnet_b1','efficientnet_b7']
embeddings_to_combine = args.embeddings_to_combine

#graph_name_opts = ['all_neigs_graph','top5_graph','top5_graph_unnorm']
#graph_name_opts = ['slicem_edge_list']
#graph_name_opts = ['slicem_edge_list_l1']
#graph_name_opts = ['slicem_edge_list_euclidean','slicem_edge_list_l1']
#graph_name_opts = ['slicem_edge_list_cosine']
#graph_name_opts = ['slicem_edge_list_l2']
graph_name_opts = args.graph_name_opts

#graph_types = ['directed','undirected']
#graph_types = ['directed']
graph_types = args.graph_types

for graph_name in graph_name_opts:
    for graph_type in graph_types:
        with open('../data/' + dataset_type + '_dataset/' + graph_name + '.txt','rb') as f:    
            if graph_type == 'directed':
                g = nx.read_weighted_edgelist(f, create_using=nx.DiGraph())
            else:
                g = nx.read_weighted_edgelist(f)
                
        #print(sorted(g.nodes()))
                
        for combined in combined_opts:
            if combined:
                # Read image node embeddings as features
                for embedding_to_combine in embeddings_to_combine:
                    if dataset_type == 'real':
                        if embedding_to_combine in ['siamese','siamese_noisy','siamese_more_projs_noisy']:
                            with open('../results/real_all/real_siamese_transferred/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)
                        elif embedding_to_combine == 'siamese_real':
                            with open('../results/real_all/real_own_siamese_0.36/siamese_real/siamese_real_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)       
                        elif embedding_to_combine == 'siamese_real_synthetic':
                            with open('../results/real_all/real_siamese_real_synthetic/siamese_real_synthetic/siamese_real_synthetic_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)                                  
                        elif embedding_to_combine == 'siamese_more_projs_all':
                            with open('../results/real_all/real_siamese_more_projs_all_efficientnet/siamese/siamese_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f) 
                        elif embedding_to_combine in ['efficientnet_b1','efficientnet_b7']:
                            with open('../results/real_all/real_siamese_more_projs_all_efficientnet/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)    
                        else:
                            with open('../results/real_all/real_original_replicate/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)
                    elif dataset_type in ['synthetic_noisy','synthetic_more_projs_noisy','synthetic_more_projs_wo_4v6c','synthetic_more_projs']: # synthetic             
                        with open('../results/'+dataset_type+'all/'+dataset_type+'__/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                            image_embeddings = np.load(f)                              
                    else: # synthetic
                        if embedding_to_combine in ['siamese_more_projs_all','efficientnet_b1','efficientnet_b7']:
                            with open('../results/synthetic_all/synthetic_big_siamese_and_efficientnet/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)   
                        else:                    
                            with open('../results/synthetic_all/synthetic_original_replicate/'+embedding_to_combine+'/'+embedding_to_combine+'_reduced_embeddings.npy', 'rb') as f:
                                image_embeddings = np.load(f)
                            
                    node_data = pd.DataFrame(image_embeddings,index=[str(num) for num in range(len(image_embeddings))])
                    #print(node_data.loc[['85']])
                    G = StellarGraph.from_networkx(g, node_features=node_data)
                    get_graph_embeddings(G, combined, embedding_to_combine,dataset_type, graph_name, graph_type)
                    
            else:
                embedding_to_combine = ''
                G = StellarGraph.from_networkx(g)
                
                get_graph_embeddings(G, combined, embedding_to_combine,dataset_type, graph_name, graph_type)
            
