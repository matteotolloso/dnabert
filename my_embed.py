from typing import Tuple
from pathlib import Path
import numpy as np
import torch
from transformers import BertModel, BertConfig, AutoTokenizer, BertTokenizerFast
import sys
import os
import time
from Bio.Seq import Seq
from Bio import SeqIO
from multiprocessing import Process


# ********* SETTINGS **********



FASTA_FILE_PATH = "../BioEmbedding/dataset/globins/globins.fasta"
OUT_DIR = "../BioEmbedding/dataset/globins/embeddings/dnabert"
MAX_CHUNK_SIZE = 510
FAST_MODE = True


DNABERT_PATH = Path("dnabert")

# ******************************


def split_sequence(seq: str, k: int=6) -> str:
    """
    Splits a sequence in a set of 6-mers and the joins it together.

    Arguments
    ---------
    seq (str): a sequence of bases.
    k (int): the length of the k-mer (defaults to 6).

    Returns
    -------
    joined_seq (str): the original string split into k-mers (separated by
    spaces)
    """
    kmers = [seq[x:x+k] for x in range(0, len(seq) + 1 - k)]
    joined_seq = " ".join(kmers)
    return joined_seq


def load_dnabert() -> Tuple[BertModel, BertTokenizerFast]:
    """
    Loads DNABert and the related tokenizer.

    Returns
    -------
    model (BertModel): the model
    tokenizer (BertTokenizerFast): the tokenizer
    """
    config = BertConfig.from_pretrained("zhihan1996/DNA_bert_6")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
    model = BertModel.from_pretrained(DNABERT_PATH, config=config)
    return model, tokenizer


def predict(id, chunk_index, query_sequence, write_to_path=False):

    start = time.time()

    device = "cpu"
    device = torch.device(device)
    model, tokenizer = load_dnabert()
    model = model.to(device)

    print(len(query_sequence), file=sys.stderr, flush=True)


    kmerized_sequence = split_sequence(query_sequence)
    
    model_inputs = tokenizer(
            kmerized_sequence,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )
    
    with torch.no_grad():
        z = model(
            input_ids=model_inputs["input_ids"].to(device),
            token_type_ids=model_inputs["token_type_ids"].to(device),
            attention_mask=model_inputs["attention_mask"].to(device),
    )
        
    z = np.array(z.last_hidden_state).reshape(-1, 768)

    if write_to_path != False:
        np.save(write_to_path, z)

    end = time.time()

    print(f"Time for embedding: {end - start}", file=sys.stderr, flush=True)

    return z


def main():

    pid = os.getpid()
    print(f'{pid}, {FASTA_FILE_PATH}', file=sys.stderr, flush=True)

    # check if the output directory exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)        

    for count, seqrecord in enumerate(SeqIO.parse(FASTA_FILE_PATH, "fasta")):

        seq_id = seqrecord.id

        # check if the file already exists
        if os.path.exists(os.path.join(OUT_DIR, f"{seq_id}.npy")):
            print(f"Skipping {seq_id} because already exists", file=sys.stderr, flush=True)
            continue

        seq_string = str(seqrecord.seq)
        seq_string = seq_string.replace(" ", "").replace("\n", "")

        
        # split the sequence in chunks such that each chunk has approximately the same length
        N = int(np.ceil(len(seq_string) / MAX_CHUNK_SIZE)) # number of chunks
        chunks = [seq_string[(i*len(seq_string))//N:((i+1)*len(seq_string))//N] for i in range(N)] # list of chunks

        lens = [len(chunk) for chunk in chunks]

        if (FAST_MODE):
            
            sequence_embedding = []
            for chunk_index, chunk in enumerate(chunks):
                print(f"Predicting the embedding {count+1}, chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr, flush=True)
                z = predict(id=seq_id, chunk_index=chunk_index, query_sequence=chunk, write_to_path=False)    
                sequence_embedding.append(z)

            # can happen that che subsequences are not of the same length, in this case pad them with the mean value
            max_len = max([len(z) for z in sequence_embedding]) # z is a np array size (chunk_size x 786)
            for i, z in enumerate(sequence_embedding):
                if len(z) < max_len:
                    sequence_embedding[i] = np.append(z, [np.mean(z, axis=0)], axis=0) # it is enough to append only one value, since the max difference between chunks is 1

            sequence_embedding = np.array(sequence_embedding)
            
            # save the embedding
            np.save(os.path.join(OUT_DIR, f"{seq_id}.npy"), sequence_embedding)
        
        else:
             
            # save each chunk in a separate file
            for chunk_index, chunk in enumerate(chunks):
                print(f"Predicting the embedding {count+1}, chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr, flush=True)
                # check if the file already exists
                if os.path.exists(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy")):
                    print(f"Skipping {seq_id}_chunk:{chunk_index+1} because already exists", file=sys.stderr, flush=True)
                    continue
                # run predict in a separate process
                p = Process(target=predict, args=(seq_id, chunk_index, chunk, os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy")))
                p.start()
                p.join()
            
            # load the chunks and recombine them
            print(f"Recombining the chunks for {seq_id}", file=sys.stderr, flush=True)
            sequence_embedding = []
            for chunk_index in range(len(chunks)):
                z = np.load(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy"))
                sequence_embedding.append(z)
                # remove the chunk file
                os.remove(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy"))

            # perform the padding as before
            max_len = max([len(z) for z in sequence_embedding]) # z is a np array size (chunk_size x 1280)
            for i, z in enumerate(sequence_embedding):
                if len(z) < max_len:
                    sequence_embedding[i] = np.append(z, [np.mean(z, axis=0)], axis=0) # it is enough to append only one value, since the max difference between chunks is 1

            sequence_embedding = np.array(sequence_embedding)
            
            # save the embedding
            np.save(os.path.join(OUT_DIR, f"{seq_id}.npy"), sequence_embedding)


    print(f'{pid}, DONE', file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
