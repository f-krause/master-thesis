import os
import time
import logging
import pickle
import json
from tqdm import tqdm
from ViennaRNA import fold
import linearfold

# FOLD_PACKAGE = 'viennarna'
FOLD_PACKAGE = 'linearfold'
DATA_PATH = '/mnt/data/krausef99dm_thesis/data'

OVERWRITE_FILES = False  # only for debugging!
MAX_SEQ_LENGTH = 2000  # 3241 seq with len below 2000
MAX_PRED_NR = 600

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        # add time stamp to log file
                        logging.FileHandler(f"logs/{FOLD_PACKAGE}_pred_sec_struc_{time.strftime('%Y%m%d-%H%M')}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()


def _load_data():
    with open(os.path.join(DATA_PATH, 'ptr_data.pkl'), 'rb') as f:
        # TODO make sure to drop super long sequences from data!
        data = pickle.load(f)
    return data


def _check_file_exists(idx):
    path = os.path.join(DATA_PATH, "sec_struc", f'{idx}-{FOLD_PACKAGE}.json')
    if os.path.exists(path):
        logging.warning(f"Skipping {idx}. File {path} already exists.")
        return True
    return False


def _store_preds(idx, preds):
    path = os.path.join(DATA_PATH, "sec_struc", f'{idx}-{FOLD_PACKAGE}.json')
    # store as json
    with open(path, 'w') as f:
        json.dump(preds, f)


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


def main():
    logging.info("Predicting secondary structure and loop type.")
    logging.info(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    logging.info(f"MAX_PRED_NR: {MAX_PRED_NR}")
    logging.info(f"FOLD_PACKAGE: {FOLD_PACKAGE}")

    data = _load_data()
    ids = data.keys()

    too_long_seq = []
    counter = 0
    start_time = time.time()

    # for idx in ["ENST00000304312"]:  # dev
    for idx in tqdm(ids):
        preds = {}
        if counter >= MAX_PRED_NR:
            break
        if not OVERWRITE_FILES and _check_file_exists(idx):
            continue

        seq = data[idx]['fasta']

        if len(seq) > MAX_SEQ_LENGTH:
            logging.warning(f"Skipping {idx}. Sequence longer than {MAX_SEQ_LENGTH}.")
            too_long_seq.append(idx)
            continue

        logging.info(f"Computing predictions for: {idx}-{FOLD_PACKAGE}")

        # Predicting structure
        if FOLD_PACKAGE == 'viennarna':
            pred_struc, mfe = fold(seq)
        elif FOLD_PACKAGE == 'linearfold':
            pred_struc, mfe = linearfold.fold(seq)
        else:
            raise ValueError(f"Unknown FOLD_PACKAGE: {FOLD_PACKAGE}")

        pred_loop_type = _pred_loop_type(seq, pred_struc)

        preds["structure"] = pred_struc
        preds["loop_type"] = pred_loop_type[5]
        preds["MFE"] = mfe
        preds["other_bpRNA_output"] = pred_loop_type[7:]

        _store_preds(idx, preds)
        counter += 1

    logging.info(f"Skipped {len(too_long_seq)} sequences because they were too long.")
    logging.info("Sequences that were skipped: ")
    logging.info(too_long_seq)

    logging.info(f"Predicted {counter} secondary structures in {(time.time() - start_time)/60} minutes.")
    logging.info(f"Average time per prediction: {(time.time() - start_time) / counter} seconds.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
