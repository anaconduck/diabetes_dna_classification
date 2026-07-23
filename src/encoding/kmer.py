import numpy as np
import os
from gensim.models import Word2Vec

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def extract_kmers(dataset, ksize):
    kmers_list = []
    labels_list = []
    
    for index, row in dataset.iterrows():
        seq = row['sequence']
        label = row['class']
        kmer = build_kmers(seq, ksize)
        kmers_list.append(kmer)
        labels_list.append(label)
        
    return kmers_list, labels_list

def extract_kmers_with_padding(dataset, ksize):
    kmers_list, labels_list = extract_kmers(dataset, ksize)
    
    max_len = 0
    for kmers in kmers_list:
        if len(kmers) > max_len:
            max_len = len(kmers)
            
    padded_kmers_list = []
    for kmers in kmers_list:
        pad_length = max_len - len(kmers)
        padded = kmers + ['[PAD]'] * pad_length
        padded_kmers_list.append(padded)
        
    return (padded_kmers_list, max_len), labels_list

def train_or_load_word2vec(kmers_train, config, save_dir="weights"):
    min_count = config['encoding']['word2vec']['min_count']
    window = config['encoding']['word2vec']['window']
    vector_size = config['encoding']['word2vec']['vector_size']
    workers = config['encoding']['word2vec']['workers']
    
    w2v_model = Word2Vec(kmers_train, 
                         min_count=min_count, 
                         window=window, 
                         vector_size=vector_size, 
                         workers=workers)
    
    os.makedirs(save_dir, exist_ok=True)
    w2v_model.save(os.path.join(save_dir, "word2vec.model"))
    
    return w2v_model

def create_seqs_embedding(w2v_model, seqs, vector_size):
    seqs_embedding = np.array([np.zeros(vector_size)])
    for seq in seqs:
        vector = np.zeros(vector_size)
        for word in seq:
            # Handle OOV words safely just in case
            if word in w2v_model.wv:
                vector += w2v_model.wv[word]
        number_of_word_in_seq = len(seq)
        if number_of_word_in_seq > 0:
            normalized_vector = vector / number_of_word_in_seq
        else:
            normalized_vector = vector
        seqs_embedding = np.append(seqs_embedding, [normalized_vector], axis=0)

    return seqs_embedding[1:]

def create_seqs_sequence_embedding(w2v_model, seqs, vector_size, max_len):
    num_samples = len(seqs)
    tensor = np.zeros((num_samples, max_len, vector_size))
    
    for i, seq in enumerate(seqs):
        for j, word in enumerate(seq):
            if word == '[PAD]':
                continue # leave as zero vector
            if word in w2v_model.wv:
                tensor[i, j, :] = w2v_model.wv[word]
                
    return tensor
