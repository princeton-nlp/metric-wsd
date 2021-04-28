from pathlib import Path


class Config:
    # run
    seed = 0
    PROJECT_NAME = 'metric-wsd'

    # params
    BERT_DIM = 768

    # paths
    ROOT_DIR = Path('path/to/your/repo')
    PROJECT_DIR = ROOT_DIR / PROJECT_NAME
    EXP_DIR = PROJECT_DIR / 'experiments'
    TMP_DIR = PROJECT_DIR / 'tmp'

    DATA_DIR = PROJECT_DIR / 'data'
    WSD_DATA_DIR = DATA_DIR / 'WSD_Evaluation_Framework'

    SCORER_DIR = WSD_DATA_DIR / 'Evaluation_Datasets'
    SEMCOR = (WSD_DATA_DIR / 'Training_Corpora/SemCor', 'semcor')
    SE07 = (WSD_DATA_DIR / 'Evaluation_Datasets/semeval2007', 'semeval2007')
    ALL = (WSD_DATA_DIR / 'Evaluation_Datasets/ALL', 'ALL')

    WN = WSD_DATA_DIR / 'Data_Validation/candidatesWN30.txt'
    GLOSS = WSD_DATA_DIR / 'Data_Validation/index.sense.gloss'

    SE07_GOLD_KEY_PATH = SE07[0] / f'{SE07[1]}.gold.key.txt'
    ALL_GOLD_KEY_PATH = ALL[0] / f'{ALL[1]}.gold.key.txt'

    pos_map = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}