import numpy as np
import pandas as pd
import random
from tabulate import tabulate

import hash_utils
import audio_signals_utils

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30 # TODO: to be tuned!
THRESHOLD = 0 # TODO: to be tuned!

def extract_shingle_dataframe(file_name, track_name, track_artist=""):
    '''
    Extracts a Pandas Dataframe having audio peaks as indexes and 1 as values.
    Params:
        [file_name]     : used by our audio signals utilities to find audio peaks
        [track_name]    : track_name that is used together with [track_artist] to create column name
        [track_artist]  : track_artist that is used together with [track_name] to create column name
    '''
    # extract audio peaks
    track, sr, onset_env, peaks = audio_signals_utils.load_audio_peaks(file_name, OFFSET, DURATION, HOP_SIZE)

    indexes = []
    # rounds peaks values to third decimal
    for el in onset_env[peaks] :
        idx = str(np.round(el, 3))
        if idx not in indexes :
            indexes.append(idx)
    # return a dataframe containing our peaks
    return pd.DataFrame(1, index=indexes, columns=["{}{}".format(f"{track_artist} - " if track_artist != "" else "", track_name)])

def extract_track_info_and_shingle_dataframe(file_name) :
    '''
    Extracts a Pandas Dataframe having audio peaks as indexes and 1 as values.    
    This makes use of [extract_shingle_dataframe] to compute the dataframe.    
    Params:
        [file_name]     : used by our audio signals utilities to find audio peaks
    '''
    track_name = file_name.stem
    track_artist = str(file_name).split('\\')[2]
    return extract_shingle_dataframe(file_name, track_name, track_artist)

def generate_hash_vector(input_set) :
    '''
    performs a random shuffle of the [input_set], returning a random hash permutation order list.
    Params:
        [input_set] : Set to shuffle
    '''
    hash_set = input_set
    random.shuffle(hash_set)
    return hash_set

# generate our series of hashFunctions 
def generate_hash_functions(amount, seed, shingle_mat):
    '''
    Generates random hash functions based on params using a seed for replicability
    Params:
        [amount]        : amount of functions to generate
        [seed]          : Random seed used to have the same functions
        [shingle_mat]   : Matrix on which to perform permutations hash functions
    '''
    hash_functions = []
    random.seed(seed)
    # randomly shuffle set to define a function
    for i in range(amount) :
        hash_functions.append(generate_hash_vector(list(shingle_mat.index)))
    return hash_functions


def extract_min_hash_signature(series, hash_functions):
    '''
    Extracts a min hash signature of a given [series].
    Params:
        [series]            : Pandas dataframe on which to apply the hash_functions
        [hash_functions]    : Functions that generate signature components
    '''
    try:
        signature = []
        for hash in hash_functions:
            # here we iterate through the indexes to see where our first match is
            for row_index in hash:
                idx = hash.index(row_index)
                # the first index having 1 as value will be our min hash
                if series.loc[row_index].item() == 1:
                    signature.append(idx)
                    break
    except Exception as e:
        print(f"{e} on {series.columns[0]}")
    return signature

def add_sig_to_dict(col, dict, hash_functions, matrix) :
    '''
    Extracts min hash signatures and adds given result to a dictionary, mapping the corrispondent column name as key.
    Params:
        [col]               : column on which to extract signature
        [dict]              : output dictionary
        [hash_functions]    : functions with which to extract signatures
        [matrix]            : input matrix on which to extract signatures
    '''
    sig = extract_min_hash_signature(matrix[col], hash_functions)
    dict[col] = sig

def convert_signatures_to_buckets_matrix(df, b, r) :
    '''
    Takes on a dataframe and divides it into bands, returns a buckets matrix. 
    Params:
        [df]    : input dataframe
        [b]     : amount of bands
        [r]     : amount of rows per band
    '''
    # Break our matrix into bands
    banded_sig = np.split(df, b)

    # column vector to convert binary vector to integer e.g. (1,0,1)->5
    binary_column = 2**np.arange(r)

    # build a buckets matrix
    buckets_matrix = []
    for sig_portion in banded_sig:
        buckets_matrix.append(np.hstack([sig_portion[column]@binary_column for column in sig_portion ] ))

    return np.vstack(buckets_matrix)


def perform_query(file_name, random_seed, hash_amount, b, r, buckets_to_songs_dict, threshold, matrix) :
    '''
    Perform a query and return best match results
    Params:
        [file_name]             : query file on which to find matches
        [random_seed]           : seed to extract hash functions
        [hash_amount]           : hash functions amount
        [b]                     : amount of bands in the signature matrix
        [r]                     : amount of rows per band
        [buckets_to_songs_dict] : dictionary mapping buckets to songs
        [threshold]             : threshold on which to decide what results are accepted
        [matrix]                : referral matrix on which to build query shingles matrix
    '''
    track_name = file_name.stem

    # initialize full shingles matrix
    query_shingles = pd.DataFrame(0, index = matrix.index, columns=[track_name])
    # extract query tracks metrix
    partials_shingles = extract_shingle_dataframe(file_name, track_name)
    # construct our query shingles
    for index in partials_shingles.index :
        query_shingles.loc[float(index)] = partials_shingles.loc[index]

    # set the seed to make sure functions are the same as the datasets
    random.seed(random_seed)
    # generate hash functions
    hash_functions = generate_hash_functions(hash_amount, random_seed, query_shingles)
    # extract our query signatures
    signature = pd.DataFrame(data = {track_name : extract_min_hash_signature(query_shingles, hash_functions)})
    # convert our signature into a single column buckets matrix
    b_col = convert_signatures_to_buckets_matrix(signature, b, r).reshape(1, -1)[0]

    # start looking for results
    result_dict = {}
    for el in b_col :
        idx = (int(el))
        matches = buckets_to_songs_dict[idx]
        for match in matches :
            # if a song gets a match, save how many times it matched
            if (match in result_dict.keys()) :
                result_dict[match] += 1
            else :
                result_dict[match] = 1
    # compute scores
    scores_dict = {}
    for key in result_dict.keys() :
        score = result_dict[key]/b
        if score >= threshold :
            scores_dict[key] = score
    # heap sort scores
    top_scores = hash_utils.heap_sort_scores(scores_dict)
    # pretty print scores
    print(f"Best results for your query on {track_name} are... ")
    headers = ['Song', 'Score']
    print(tabulate(top_scores.items(), headers=headers, tablefmt="pretty"))
    return top_scores
    
    