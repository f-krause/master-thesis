TOKENS = "01235ACGT().BEHIMSX"

TISSUES = ['Adrenal', 'Appendices', 'Brain', 'Colon', 'Duodenum', 'Uterus',
           'Esophagus', 'Fallopiantube', 'Fat', 'Gallbladder', 'Heart', 'Kidney',
           'Liver', 'Lung', 'Lymphnode', 'Ovary', 'Pancreas', 'Placenta',
           'Prostate', 'Rectum', 'Salivarygland', 'Smallintestine', 'Smoothmuscle',
           'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Tonsil', 'Urinarybladder']

# Note: per convention, mRNA is encoded as DNA, although T in DNA is replaced by U in mRNA
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