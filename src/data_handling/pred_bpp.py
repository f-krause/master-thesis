# For BPPs: https://github.com/DasLab/arnie.git
import os.path

from dotenv import load_dotenv

load_dotenv()

import os
import pickle
import json
import logging
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix
from arnie.bpps import bpps
from arnie.mea.mea import MEA

FOLD_PACKAGE = 'vienna_2'  # see doc for more packages: https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md
DATA_PATH = '/export/share/krausef99dm/data'

OVERWRITE_FILES = True  # FIXME for dev only
MAX_SEQ_LENGTH = 400  # 3241 seq with len below 2000
MAX_PRED_NR = 2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        # add time stamp to log file
                        logging.FileHandler(f"logs/pred_bpp_{time.strftime('%Y%m%d-%H%M')}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()


def _load_data():
    with open(os.path.join(DATA_PATH, 'ptr_data.pkl'), 'rb') as f:
        # TODO make sure to drop super long sequences from data!
        data = pickle.load(f)
    return data


# Compute BPPs
def _pred_bpps(idx, seq):
    try:
        # data[transcript]['fasta']
        return bpps(seq, package=FOLD_PACKAGE)
    except Exception as e:
        logger.error(f"{idx}: Could not compute BPPs with package {FOLD_PACKAGE}.")
        logger.error(e)


def _pred_structure(bpp_matrix):
    return MEA(bpp_matrix).structure


def _pred_loop_type(seq, structure, debug=False):
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


def _check_file_exists(idx):
    path = os.path.join(DATA_PATH, "bpp", f'{idx}-{FOLD_PACKAGE}.json')
    if os.path.exists(path):
        logging.warning(f"Skipping {idx}. File {path} already exists.")
        return True
    return False


def _store_preds(idx, preds):
    path = os.path.join(DATA_PATH, "bpp", f'{idx}-{FOLD_PACKAGE}.json')
    with open(path, 'wb') as f:
        pickle.dump(preds, f)


def main():
    logging.info("Predicting secondary structure and loop type.")

    data = _load_data()
    ids = data.keys()

    counter = 0
    start_time = time.time()

    # for idx in ["ENST00000304312"]:  # FIXME dev
    for idx in tqdm(ids):
        preds = {}
        if counter >= MAX_PRED_NR:
            break
        if not OVERWRITE_FILES and _check_file_exists(idx):
            continue

        seq = data[idx]['fasta']

        if len(seq) > MAX_SEQ_LENGTH:
            # logging.warning(f"Skipping {idx}. Sequence longer than {MAX_SEQ_LENGTH}.")  # FIXME for dev only
            continue

        logging.info(f"Computing predictions for: {idx}-{FOLD_PACKAGE}")
        logging.info(f"   Sequence length: {len(seq)}")

        start_time_seq = time.time()
        bpp_temp = _pred_bpps(idx, seq)

        preds['structure'] = _pred_structure(bpp_temp)

        preds['loop_type'] = _pred_loop_type(seq, preds['structure'])

        # store bpp as sparse matrix
        bpp_temp[bpp_temp < 0.001] = 0  # set all pairing probabilities below 0.001 to 0
        bpp_temp_sparse = csr_matrix(bpp_temp)

        preds['bpp'] = bpp_temp_sparse

        logging.info(f"   Pred time: {(time.time() - start_time_seq)} seconds.")

        _store_preds(idx, preds)
        counter += 1

    logging.info(f"Predicted {counter} secondary structures in {(time.time() - start_time)/60} minutes.")
    logging.info(f"Average time per prediction: {(time.time() - start_time) / counter} seconds.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
