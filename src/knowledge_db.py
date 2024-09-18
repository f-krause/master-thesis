TISSUES = ['Adrenal', 'Appendices', 'Brain', 'Colon', 'Duodenum', 'Uterus',
           'Esophagus', 'Fallopiantube', 'Fat', 'Gallbladder', 'Heart', 'Kidney',
           'Liver', 'Lung', 'Lymphnode', 'Ovary', 'Pancreas', 'Placenta',
           'Prostate', 'Rectum', 'Salivarygland', 'Smallintestine', 'Smoothmuscle',
           'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Tonsil', 'Urinarybladder']

CODON_MAP = {
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
