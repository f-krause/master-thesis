TISSUES = ['Adrenal', 'Appendices', 'Brain', 'Colon', 'Duodenum', 'Uterus',
           'Esophagus', 'Fallopiantube', 'Fat', 'Gallbladder', 'Heart', 'Kidney',
           'Liver', 'Lung', 'Lymphnode', 'Ovary', 'Pancreas', 'Placenta',
           'Prostate', 'Rectum', 'Salivarygland', 'Smallintestine', 'Smoothmuscle',
           'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Tonsil', 'Urinarybladder']

CODON_MAP_RNA = {
    # Phenylalanine (Phe)
    'UUU': 1, 'UUC': 2,
    # Leucine (Leu)
    'UUA': 3, 'UUG': 4, 'CUU': 5, 'CUC': 6, 'CUA': 7, 'CUG': 8,
    # Isoleucine (Ile)
    'AUU': 9, 'AUC': 10, 'AUA': 11,
    # Methionine (Met) (Start codon)
    'AUG': 12,
    # Valine (Val)
    'GUU': 13, 'GUC': 14, 'GUA': 15, 'GUG': 16,
    # Serine (Ser)
    'UCU': 17, 'UCC': 18, 'UCA': 19, 'UCG': 20, 'AGU': 21, 'AGC': 22,
    # Proline (Pro)
    'CCU': 23, 'CCC': 24, 'CCA': 25, 'CCG': 26,
    # Threonine (Thr)
    'ACU': 27, 'ACC': 28, 'ACA': 29, 'ACG': 30,
    # Alanine (Ala)
    'GCU': 31, 'GCC': 32, 'GCA': 33, 'GCG': 34,
    # Tyrosine (Tyr)
    'UAU': 35, 'UAC': 36,
    # Stop codons
    'UAA': 37, 'UAG': 38, 'UGA': 39,
    # Histidine (His)
    'CAU': 40, 'CAC': 41,
    # Glutamine (Gln)
    'CAA': 42, 'CAG': 43,
    # Asparagine (Asn)
    'AAU': 44, 'AAC': 45,
    # Lysine (Lys)
    'AAA': 46, 'AAG': 47,
    # Aspartic acid (Asp)
    'GAU': 48, 'GAC': 49,
    # Glutamic acid (Glu)
    'GAA': 50, 'GAG': 51,
    # Cysteine (Cys)
    'UGU': 52, 'UGC': 53,
    # Tryptophan (Trp)
    'UGG': 54,
    # Arginine (Arg)
    'CGU': 55, 'CGC': 56, 'CGA': 57, 'CGG': 58, 'AGA': 59, 'AGG': 60,
    # Glycine (Gly)
    'GGU': 61, 'GGC': 62, 'GGA': 63, 'GGG': 64
}

# FIXME: why is there T? Shouldn't it be U for mRNA?
CODON_MAP_DNA = {
    # Phenylalanine (Phe)
    'TTT': 1, 'TTC': 2,
    # Leucine (Leu)
    'TTA': 3, 'TTG': 4, 'CTT': 5, 'CTC': 6, 'CTA': 7, 'CTG': 8,
    # Isoleucine (Ile)
    'ATT': 9, 'ATC': 10, 'ATA': 11,
    # Methionine (Met) (Start codon)
    'ATG': 12,
    # Valine (Val)
    'GTT': 13, 'GTC': 14, 'GTA': 15, 'GTG': 16,
    # Serine (Ser)
    'TCT': 17, 'TCC': 18, 'TCA': 19, 'TCG': 20, 'AGT': 21, 'AGC': 22,
    # Proline (Pro)
    'CCT': 23, 'CCC': 24, 'CCA': 25, 'CCG': 26,
    # Threonine (Thr)
    'ACT': 27, 'ACC': 28, 'ACA': 29, 'ACG': 30,
    # Alanine (Ala)
    'GCT': 31, 'GCC': 32, 'GCA': 33, 'GCG': 34,
    # Tyrosine (Tyr)
    'TAT': 35, 'TAC': 36,
    # Stop codons
    'TAA': 37, 'TAG': 38, 'TGA': 39,
    # Histidine (His)
    'CAT': 40, 'CAC': 41,
    # Glutamine (Gln)
    'CAA': 42, 'CAG': 43,
    # Asparagine (Asn)
    'AAT': 44, 'AAC': 45,
    # Lysine (Lys)
    'AAA': 46, 'AAG': 47,
    # Aspartic acid (Asp)
    'GAT': 48, 'GAC': 49,
    # Glutamic acid (Glu)
    'GAA': 50, 'GAG': 51,
    # Cysteine (Cys)
    'TGT': 52, 'TGC': 53,
    # Tryptophan (Trp)
    'TGG': 54,
    # Arginine (Arg)
    'CGT': 55, 'CGC': 56, 'CGA': 57, 'CGG': 58, 'AGA': 59, 'AGG': 60,
    # Glycine (Gly)
    'GGT': 61, 'GGC': 62, 'GGA': 63, 'GGG': 64
}