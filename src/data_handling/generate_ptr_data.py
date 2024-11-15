"""
Creates labeled sequences for processing.

Code shared by Prof. Sebastian Tschiatschek

ptr.csv - Sheet 2 from Additional File 2 from Hernandez-Alias (Base: Human Proteine Atlas)
link: https://figshare.com/articles/dataset/Additional_file_2_Protein-to-mRNA_ratios_among_tissues/21379197?file=37938894

Options:
- codon frequencies
- only codon region
- discretization
"""

import os
import platform
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from collections import OrderedDict
from Bio import SeqIO

if platform.node() == "rey" or platform.node() == "jyn":
    BED_FILES_FOLDER = "/export/share/krausef99dm/data/ptr_data/BED6__protein_coding_strict"
    FASTA_FILES_FOLDER = "/export/share/krausef99dm/data/ptr_data/FA_protein_coding_strict_mRNA"
    PTR_CSV_FILE = "/export/share/krausef99dm/data/ptr_data/ptr.csv"
else:
    BED_FILES_FOLDER = "../../data/ptr_data/BED6__protein_coding_strict"
    FASTA_FILES_FOLDER = "../../data/ptr_data/FA_protein_coding_strict_mRNA"
    PTR_CSV_FILE = "../../data/ptr_data/ptr.csv"

TISSUES = ['Adrenal', 'Appendices', 'Brain', 'Colon', 'Duodenum', 'Uterus',
           'Esophagus', 'Fallopiantube', 'Fat', 'Gallbladder', 'Heart', 'Kidney',
           'Liver', 'Lung', 'Lymphnode', 'Ovary', 'Pancreas', 'Placenta',
           'Prostate', 'Rectum', 'Salivarygland', 'Smallintestine', 'Smoothmuscle',
           'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Tonsil', 'Urinarybladder']
LOAD_LIMIT = 2
# For limit 1000 loaded only 419 in the end? What is wrong with the other fasta files?
# For all files loaded 11.279 sequences

base_to_int = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

int_to_base = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T'
}


def _seq_to_int(seq):
    """
    Convert a sequence of base pairs to integers.
    """
    seq = np.array(list(map(lambda x: base_to_int[x], seq)), dtype=np.int8)
    return seq


def _seq_to_frequency(seq, bed):
    """
    embed the sequence into a vector of codon frequencies
    """
    # find codon region
    idx = np.where([p[3] == 'CDS' for p in bed])[0][0]
    start = int(bed[idx][1])
    end = int(bed[idx][2])

    # compute codon frequencies
    codons = np.sum(seq[start:end].reshape(-1, 3) * np.array([[4 ** 0, 4 ** 1, 4 ** 2]]), axis=1)
    counts = np.histogram(codons, range(65))[0]
    freq = counts / np.sum(counts)

    return freq


def read_bed_file(filename):
    with open(filename, 'rt') as f:
        lines = f.readlines()
        data = [l.split() for l in lines]

    return data


def _annotate_bed(content):
    seq_length = np.max([int(line[2]) for line in content])
    out = np.zeros(seq_length)
    for line in content:
        start = int(line[1])
        end = int(line[2])
        if line[3] == '5UTR':
            out[start:end] = 5
        elif line[3] == '3UTR':
            out[start:end] = 3
        elif line[3] == 'CDS':
            if (end - start) % 3 > 0:
                logging.error(f"line: {line}")
            assert (end - start) % 3 == 0, "Invalid CDS length."
            out[start:end] = np.array([[0, 1, 2]]).repeat((end - start) // 3, axis=0).ravel()
        else:
            logging.error(f"Invalid type: {line[3]}")
            raise ValueError(f"Invalid type: {line[3]}")

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ## (0) read in PTRs
    df = pd.read_csv(PTR_CSV_FILE)

    ## (1) preparse dict of FASTA/BED files with same transcript ID
    transcripts = df['EnsemblTranscriptID'].to_list()

    PTRs = {}
    # for i in tqdm(range(len(df))):
    #     transcript = df.iloc[i]['EnsemblTranscriptID']
    #     PTRs[transcript] = df.iloc[i][TISSUES].to_numpy()
    for transcript, data in zip(df['EnsemblTranscriptID'].to_numpy(), df[TISSUES].to_numpy()):
        PTRs[transcript] = data

    # convert list of bed files to only the transcription name
    _bed_files = sorted(os.listdir(BED_FILES_FOLDER))
    bed_files = {}
    counter = 0
    for file in _bed_files:
        try:
            tokens = file.split('.')
            if tokens[-1] == 'bed':
                bed_files[tokens[0]] = file
            counter += 1
        except:
            logging.warning(f"Can't process filename {file}.")
        if counter >= LOAD_LIMIT:
            break


    logging.info(f"Considering {len(bed_files)} .bed files")


    _fasta_files = sorted(os.listdir(FASTA_FILES_FOLDER))
    fasta_files = {}
    counter = 0
    for file in _fasta_files:
        try:
            tokens = file.split('.')
            if tokens[-1] == 'fasta':
                fasta_files[tokens[0]] = file
            counter += 1
        except:
            logging.warning(f"Can't process filename {file}.")
        if counter >= LOAD_LIMIT:
            break

    logging.info(f"Considering {len(fasta_files)} .fasta files")

    ## (2) 
    data = OrderedDict()
    for transcript in tqdm(transcripts):
        data[transcript] = {}

        # get sequence
        try:
            fasta_file = os.path.join(FASTA_FILES_FOLDER, fasta_files[transcript])
            with open(fasta_file) as f:
                content = list(SeqIO.parse(f, "fasta"))
                data[transcript]['fasta'] = str(content[0].seq)
                data[transcript]['fasta_ohe'] = _seq_to_int(str(content[0].seq))
        except Exception as e:
            logging.warning(f"Could not read fasta file for {transcript}. Error: {e}")
            del data[transcript]
            continue

        # get 5UTR and 3UTR info
        try:
            bed_file = os.path.join(BED_FILES_FOLDER, bed_files[transcript])
            data[transcript]['bed'] = read_bed_file(bed_file)
            data[transcript]['bed_annotation'] = _annotate_bed(data[transcript]['bed'])
        except:
            logging.warning(f"Could not read bed file for {transcript}.")
            del data[transcript]
            continue

        # compute codon_frequencies
        try:
            data[transcript]['codon_freq'] = _seq_to_frequency(data[transcript]['fasta_ohe'], data[transcript]['bed'])
        except:
            logging.warning(f"Could not compute vector of codon frequencies for {transcript}.")
            del data[transcript]
            continue

        # Add target data
        data[transcript]['targets'] = PTRs[transcript]

    ## compute discrete labels
    keys = data.keys()
    n_tissues = len(TISSUES)
    all_targets = np.vstack([data[t]['targets'].reshape(1, -1) for t in keys])
    max_others = []
    min_others = []
    avg_others = []
    n_others = []
    n_fold_glob = 1
    n_fold_avg = 2
    for tidx, tissue in enumerate(TISSUES):
        idx = list(range(n_tissues))
        del idx[tidx]

        max_others.append(np.nanmax(all_targets[:, idx], axis=1))
        min_others.append(np.nanmin(all_targets[:, idx], axis=1))
        avg_others.append(np.nanmean(all_targets[:, idx], axis=1))
        n_others.append(np.sum(np.logical_not(np.isnan(all_targets[:, idx])), axis=1))

    all_targets_bin = np.nan * np.ones_like(all_targets)
    for tidx, tissue in enumerate(TISSUES):
        isUP = ((all_targets[:, tidx] - max_others[tidx]) >= np.log10(n_fold_glob)) & \
               ((all_targets[:, tidx] - avg_others[tidx]) >= np.log10(n_fold_avg)) & \
               (n_others[tidx] >= 3)
        isDOWN = ((all_targets[:, tidx] - min_others[tidx]) <= -np.log10(n_fold_glob)) & \
                 ((all_targets[:, tidx] - avg_others[tidx]) <= -np.log10(n_fold_avg)) & \
                 (n_others[tidx] >= 3)

        all_targets_bin[isUP, tidx] = 1  # up = 1 -> high PTR
        all_targets_bin[isDOWN, tidx] = 0  # down = 0 -> low PTR

        logging.info(f"There are {np.sum(isUP)} ups and {np.sum(isDOWN)} downs.")

    # put labels back
    for idx, transcript in enumerate(keys):
        data[transcript]['targets_bin'] = all_targets_bin[idx, :]

    logging.info(f"Considering {len(data)} sequences...")


    # save data
    import pickle
    try:
        pickle.dump(data, open('../../data/ptr_data/ptr_data_TEST.pkl', 'wb'))
    except:
        logging.error("Could not save data.pkl.")


    # import IPython
    # IPython.embed()
