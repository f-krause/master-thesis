# For BPPs: https://github.com/DasLab/arnie.git
import os.path

from dotenv import load_dotenv

load_dotenv()

import os
import pickle
import logging
from tqdm import tqdm
from scipy.sparse import csr_matrix
from arnie.bpps import bpps
from arnie.mea.mea import MEA

FOLD_PACKAGES = ['vienna_2']  # see doc for more packages: https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md
SEC_STRUC_PATH = '/mnt/data/krausef99dm_thesis/data/sec_struc/'

OVERWRITE_FILES = True  # FIXME for dev only
PRED_BPP = True
PRED_STRUCTURE = False
PRED_LOOP_TYPE = False

logging.FileHandler("pred_sec_struc.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# Compute BPPs
def _pred_bpps(idx, folding_package, seq):
    try:
        # data[transcript]['fasta']
        return bpps(seq, package=folding_package)
    except Exception as e:
        logger.error(f"{idx}: Could not compute BPPs with package {folding_package}.")
        logger.error(e)


def _pred_structure(bpp_matrix):
    return MEA(bpp_matrix).structure


def _pred_loop_type(seq, structure, debug=False):
    # return None
    # os.system("cd folding_algorithms/bpRNA")
    os.chdir("folding_algorithms/bpRNA")
    os.system(f'echo {seq} > a.dbn')
    os.system('echo \"{}\" >> a.dbn'.format(structure))
    os.system('perl bpRNA.pl a.dbn')
    loop_type = [l.strip('\n') for l in open('a.st')]
    if debug:
        print(seq)
        print(structure)
        print(loop_type[5])
    os.chdir("/export/home/krausef99dm/master-thesis/src/data_handling")
    return loop_type


def _check_file_exists(idx, fold_package):
    path = os.path.join(SEC_STRUC_PATH, f'{idx}-{fold_package}.pkl')
    if os.path.exists(path):
        logging.warning(f"Skipping {idx}. File {path} already exists.")
        return True
    return False


def _store_preds(idx, fold_package, preds):
    path = os.path.join(SEC_STRUC_PATH, f'{idx}-{fold_package}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(preds, f)


def main():
    logging.info("Predicting secondary structure and loop type.")

    with open('/mnt/data/krausef99dm_thesis/data/ptr_data.pkl', 'rb') as f:
        # TODO make sure to drop super long sequences from data!
        data = pickle.load(f)

    ids = data.keys()

    # for idx in tqdm(ids):
    for idx in ["ENST00000304312"]:  # FIXME dev
        preds = {fold_package: {'bpp': None, 'structure': None, 'loop_type': None} for fold_package in FOLD_PACKAGES}
        seq = data[idx]['fasta']

        for fold_package in FOLD_PACKAGES:
            if _check_file_exists(idx, fold_package) and not OVERWRITE_FILES:
                continue

            logging.info(f"Computing predictions for: {idx}-{fold_package}")

            bpp_temp = _pred_bpps(idx, fold_package, seq)

            preds[fold_package]['structure'] = _pred_structure(bpp_temp)

            preds[fold_package]['loop_type'] = _pred_loop_type(seq, preds[fold_package]['structure'])

            # store bpp as sparse matrix

            bpp_temp[bpp_temp < 0.001] = 0  # set all pairing probabilities below 0.001 to 0
            bpp_temp_sparse = csr_matrix(bpp_temp)

            preds[fold_package]['bpp'] = bpp_temp_sparse

            _store_preds(idx, fold_package, preds)
        break


if __name__ == "__main__":
    main()
