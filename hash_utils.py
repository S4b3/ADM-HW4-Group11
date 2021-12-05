import random
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import json 
import re
import heapq
from tabulate import tabulate
import numpy as np
import audio_signals_utils

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30 # TODO: to be tuned!
THRESHOLD = 0 # TODO: to be tuned!


def genRandomCoefficients(k, upperBound):
    '''Generate a list of 'k' random coefficients for the random hash functions,
    while ensuring that the same value does not appear multiple times in the 
    list.
    '''
    # Create a list of 'k' random values.
    output = np.array([])

    while k > 0:
        # Get a random shingle ID.
        ranValue = random.randint(0, upperBound) 

        # Ensure that each random number is unique.
        while ranValue in output:
            ranValue = random.randint(0, upperBound) 

        # Add the random number to the list.
        output = np.append(output, ranValue)
        k = k - 1
    return output


def generateHashCoefficients(hashAmount, hashUpperBound):
    ''' 
    This returns a matrix of random coefficients for the hashFunction :   
        h(x) = (a*x + b) % c
    In the form of 
        A : [colA, colB]\n
    Parameters:
        [hashAmount] : is the amount of components to generate, more easily,
            the length of the columns
        [hashUpperBound] : is the max value a coefficient can take. For reference,
            remember that in the above formula C should be greater than this value.
    '''
    print('\nGenerating random hash function coefficients...')

    # Our random hash function will take the form of:
    #   h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a prime number just greater than hashUpperBound.

    # For each of the 'hashesAmount' hash functions, generate a different coefficient 'a' and 'b'.   
    coeffA = genRandomCoefficients(hashAmount, hashUpperBound)
    coeffB = genRandomCoefficients(hashAmount, hashUpperBound)
    return np.stack((coeffA, coeffB), axis = 1)

def customHash(x, a, b, c):   
    '''h(x) = (a*x + b) % c'''
    return (a*x + b) % c

def extractMinHashSignature(audio, hashAmount, coefficientsMatrix, nextPrimeNumberAfterUpperBound):
    '''
    Extracts MinHash Signature from audio by computing its audio peaks   
    and hashing them via the MinHash algorithm.   
    This makes use of [customHash] function.
    '''
    _, _, onset_env, peaks = audio_signals_utils.load_audio_peaks(audio, OFFSET, DURATION, HOP_SIZE)
    # initialize track signature
    trackSignature = []
    # for every hashfunction we generated, extract the minhash and add it to the signature
    for i in range(0, hashAmount) :
        hashValues = np.array([ customHash(np.round(x,3), coefficientsMatrix[i][0], coefficientsMatrix[i][1], nextPrimeNumberAfterUpperBound) for x in onset_env[peaks]])
        # Add the smallest hash value as a component of the signature.
        trackSignature.append(hashValues.min())
    # return current signature
    return trackSignature


def minHashSignatureIntoDict(file_path, dictionary, hashAmount, coefficientsMatrix, nextPrimeNumberAfterUpperBound):
    '''
    This function uses the MinHash Algorithm to compute a hash signature    
    over an audio track and save it into a dictionary.    
    Once the signature has been extracted, it gets saved inside the dictionary in position    
    audiotuple[0]

    Params:
        [audiotuple] : Tuple defined like : (trackID, audio);
        [dictionary] : Dictionary in which the signature will be saved. 
    
    '''

    file_name = re.search(r"[\w\-!$]+\.", str(file_path)).group()[:-1]

    dictionary[file_name] = extractMinHashSignature(file_path, hashAmount, coefficientsMatrix, nextPrimeNumberAfterUpperBound)

    
def heap_sort_scores(scores_dict):
    '''
    Returns the best 10 scores in a given dictionary, sorted
    '''
    tmp_dict = {}
    output = {}
    # create a score heap structure and a dictionary of scores
    heap = list()
    heapq.heapify(heap)
    
    for key,score in scores_dict.items() :
        tmp_dict[score] = key
        # Update the heap
        heapq.heappush(heap, score)
    
    # sort scores
    for score in heapq.nlargest(10, heap) :
        # find the doc associated to the score
        output[tmp_dict[score]] = score
    
    return output
    

def compute_similarity(query_sig, data_sig, num_hashes) :
    '''
    Computes two minHash signatures similarity score
    Params:
        [query_sig] : query signature
        [data_sig]  : data signature
        [num_hashes]: amount of minHashes in the signature
    '''
    similarity_count = 0
    for k in range(num_hashes) :
        similarity_count += (query_sig[k] == data_sig[k] )
    return similarity_count/num_hashes

def compute_hash_similarities(query_signature, dataset, num_hashes, threshold) :
    '''
    Computes minHash signatures similarity scores for the whole dataset    
    regarding to a query signature
    Params:
        [query_signature]   : query signature
        [dataset]           : dataset composed by minHash signatures
        [num_hashes]        : amount of minHashes in the signature
        [threshold]         : similarity threshold in respect to which results are obtained
    '''
    results_dict = {}

    def compute_sim_and_add_to_dict(query_sig, data_sig, num_hashes, track_name, dict, threshold):
        '''
        Computes minHash signatures similarity scores for two signatures
        and adds the score to a dictionary if the threshold is respected    
        Params:
            [query_sig]     : query signature
            [data_sig]      : data signature
            [num_hashes]    : amount of minHashes in the signature
            [track_name]    : name of the track we're processing, will be the dictionary's key
            [dict]          : output dictionary
            [threshold]     : similarity threshold in respect to which results are obtained
        '''
        score = compute_similarity(query_sig, data_sig, num_hashes)
        if(score >= threshold):
            dict[track_name] = score

    pool = ThreadPool(multiprocessing.cpu_count())
    result = pool.map(lambda trackTuple : compute_sim_and_add_to_dict(
        query_signature, trackTuple[1], num_hashes,  trackTuple[0], results_dict, threshold), (dataset.items()))
    return heap_sort_scores(results_dict)
    
def find_song_query(file_name, query_set, dataset, num_hashes, threshold) :
    '''
    Computes a track similarity query on a minHashSignatures dataset.
    Params:
        [file_name]         : where to find the query track
        [query_set]         : our processed signatures of every query track,
                              used to avoid recompiling signatures at every run
        [dataset]           : dataset composed by minHash signatures
        [num_hashes]        : amount of minHashes in the signature
        [threshold]         : similarity threshold in respect to which results are obtained
    
    Returns:
        Pretty prints the highest scoring tracks's title in respect to a given threshold.
    '''
    # find file : 
    track_name = re.search(r"[\w\-!$]+\.", str(file_name)).group()[:-1]
    query_signature = query_set[track_name]
    assert (query_signature != None), 'This song was not present in our query set!' 
    # results dictionary returned like 'track_name' : score
    results = compute_hash_similarities(query_signature, dataset, num_hashes, threshold)
    headers = ['Song', 'Score']
    print()
    print(f"Your query on {track_name} has produced these results : ")
    print(tabulate(results.items(), headers=headers, tablefmt="pretty"))
