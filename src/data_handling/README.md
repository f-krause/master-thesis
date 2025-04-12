# Notes on Data handling

## File Explanations
- [`data_loader.py`](data_loader.py) - contains Dataset and Dataloader definitions for pytorch based model training
- [`generate_ptr_data.py`](generate_ptr_data.py) - create the raw ptr dataset containing sequences and PTR ratios
- [`pred_sec_struc.py`](pred_sec_struc.py) - predict secondary structure for mRNA sequences and store them
- [`pred_bpp.py`](pred_bpp.py) - legacy; create bpp matrices (not recommended, as quadratic scaling)
- [`train_data_codons.py`](train_data_codons.py) - create stratified, unbiased, random data split 
- [`train_data_seq.py`](train_data_seq.py) - create data split based on split of codon level data (or random)
- [`train_val_test_indices.py`](train_val_test_indices.py) - code to perform stratified, unbiased data split


## Reproduction
Note that after creating codon and nucleotide level datasets, need to clean test dataset on codon level, s.t. both test 
sets contain the same sequences


## Notes on Data
11279 sequences, with around 24 targets per seq

### Codon data
"codon_X_2.7k_data.pkl" (X = train, test, val)

SEED = 1192  # randomly drawn with np.random.randint(0,2024) on 22.10.2024, 15:00

```
LOG: Counts per bin: 
array([ 470, 1088, 1237, 1474, 1342, 1154,  800,  685,  552,  410])  # counts
array([ 114. ,  372.3,  630.6,  888.9, 1147.2, 1405.5, 1663.8, 1922.1, 2180.4, 2438.7, 2697. ])) # bins
Num seq-tuple pairs TRAIN: 127673
Num seq-tuple pairs VAL: 27398
Num seq-tuple pairs TEST: 27554 (- 725 after alignment)
```

NOTE: to achieve high level of comparability across data, 35 sequences (i.e. 725 seq-tissue tuples) were removed from 
codon based test data, which had a CDS with at most 2.700 codons, however had a total sequence longer than 9000 
nucleotides.



### Nucleotide level data
X_2.7k_data.pkl" (X = train, test, val)
SEED = 1192 (as above) - irrelevant, because identical sequences chosen as for codon data (based on identifier, but seq longer than 9k dropped, hence a bit less)

```
Num seq-tuple pairs TRAIN: 124984
Num seq-tuple pairs VAL: 26909
Num seq-tuple pairs TEST: 26829
```

The difference to the numbers above result from the fact, that some sequences might have a CDS with less than 8.100 
nulceotides, yet the total sequence might have more than 9000 nucleotides. These sequences were removed from the test 
set, but not for the other sets. 


### Data structure
- Codon encoded sequence data
  - "codon_X_2.7k_data.pkl" (X = train, test, val) within data folders
  - max sequence length <= 2700 codons, i.e. 8100 nucleotides
  - Features:
    - rna_data: codon encoding with 64 possible codons
    - tissue_id
    - target, target_bin (classification target)
- Nucleotide level encoded data
  - "X_9.0k_data.pkl" (X = train, test, val) within data folders
  - max sequence length <= 9000 nucleotides
  - Features:
    - rna_data: nucleotide level sequence, secondary structure prediction, predicted loop type
    - tissue_id
    - target, target_bin