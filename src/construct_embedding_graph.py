import time
import argparse
import numpy as np
import multiprocessing
from scipy.stats import wasserstein_distance
from multiprocessing import cpu_count as mul_cpu_count


def main():
    """calculates similarity between 2D class averages"""
    
    parser = argparse.ArgumentParser(description='compare similarity of 2D class averages ')
    
    parser.add_argument('-i', '--input', action='store', dest='embedding', required=False,
                        default='siamese',help='path to npy file of embeddings')
    
    parser.add_argument('-o', '--outpath', action='store', dest='outpath', required=False,
                        default='../data/synthetic_dataset/',help='path for output files')
    
    parser.add_argument('-m', '--metric', action='store', dest='metric', required=False, 
                        default='EMD', choices=['Euclidean', 'L1', 'cosine', 'EMD', 'correlate'],
                        help='choose scoring method, default Euclidean')
    
    parser.add_argument('-c', '--num_workers', action='store', dest='num_workers', required=False, type=int, default=0,
                        help='number of CPUs to use, default - all cores')
           
    parser.add_argument('-t', '--translate', action='store', dest='translate', required=False, 
                        default='full', choices=['full', 'valid'],
                        help='indicate size of score vector, numpy convention, default full')
    

    args = parser.parse_args()
    
    num_cores = mul_cpu_count()
    
    if args.num_workers == 0:
        args.num_workers = num_cores
        print('No. of workers = ',args.num_workers)

    if args.metric == 'Euclidean':
        pairwise_score = pairwise_l2
    elif args.metric == 'L1':
        pairwise_score = pairwise_l1
    elif args.metric == 'cosine':
        pairwise_score = pairwise_cosine
    elif args.metric == 'EMD':
        pairwise_score = pairwise_wasserstein
    elif args.metric == 'correlate':
        pairwise_score = pairwise_correlate
                

    wrapper_function = wrapper_single_function
        
    final_scores = {}
    
    emb_name = args.embedding
    fname = '../results/synthetic_original_replicate_0.42/'+emb_name+'/'+emb_name+'_reduced_embeddings.npy'
    embeddings = np.load(fname)
    num_class_avg = len(embeddings)
    
    with multiprocessing.Pool(args.num_workers) as pool:
        for i in range(num_class_avg-1):
            line_projections_1 = dict({(i,0):(embeddings[i],i)})
            for j in range(i+1, num_class_avg):
                line_projections_2 = dict({(j,0):(embeddings[j],j)})
                           
                projection_pairs = []
                for line_1 in line_projections_1.values():
                    for line_2 in line_projections_2.values():
                        projection_pairs.append((line_1, line_2))
            
                pair_scores = pool.starmap(
                    wrapper_function, 
                    [(pair, pairwise_score, args.translate) for pair in projection_pairs]
                )

                optimum = min(pair_scores, key = lambda x: x[4])

                avg_1, deg_1, avg_2, deg_2, score = [value for value in optimum]

                final_scores[(avg_1, avg_2)] = (deg_1, deg_2, score)
                final_scores[(avg_2, avg_1)] = (deg_2, deg_1, score)
    
    write_scores(final_scores, outpath=args.outpath,metric=args.metric,emb=args.embedding)    

def pairwise_l2(a, b, translate):
    return np.linalg.norm(a - b)


def pairwise_l1(a, b, translate):
    return np.linalg.norm(a - b, 1)


def pairwise_cosine(a, b, translate):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pairwise_correlate(a, b, translate):
    s = np.correlate(a, b, mode=translate)
    return 1 / (1 + np.amax(s)) # Convert to distance


def pairwise_wasserstein(a, b, translate):
    return wasserstein_distance(a, b)

def wrapper_single_function(pair, pairwise, translate):
    score = pairwise(pair[0][0], pair[1][0], translate) 
    return [pair[0][1], 0, pair[1][1], 0, score]

                
def write_scores(final_scores, outpath, metric, emb):
    """
    tab separted file of final scores
    load scores into the slicem gui
    """
    stamp = time.strftime('%Y%m%d_%H%M%S')
    
    header = ['projection_1', 'degree_1', 'projection_2', 'degree_2', 'score']
    
    with open(outpath+'/{2}_embedding_scores_{0}_{1}.txt'.format(stamp,metric,emb), 'w') as f:
        for h in header:
            f.write(h+'\t')
        f.write('\n')
        for p, v in final_scores.items():
            f.write(str(p[0])+'\t'+str(v[0])+'\t'+str(p[1])+'\t'+str(v[1])+'\t'+str(v[2])+'\n')          

            
if __name__ == "__main__":
    starttime = time.time()
    main()
    print('Runtime: {} minutes'.format((time.time() - starttime)/60))