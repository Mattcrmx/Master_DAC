import re
import string
import unicodedata
from pathlib import Path
from tqdm import tqdm
import sentencepiece as spm


def normalize(s):
    return re.sub(' +', ' ', "".join(c if c in string.ascii_letters else " "
                                     for c in unicodedata.normalize('NFD', s.lower().strip())
                                     if c in string.ascii_letters + " " + string.punctuation)).strip()


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]


    # with open(PROJECT_DIR / 'data/en-fra.txt') as f:
    #     lines = f.readlines()
    #
    # sentences_eng = []
    # sentences_fr = []
    # for s in tqdm(lines):
    #     orig, dest = s.split("\t")[:2]
    #     sentences_eng.append(orig)
    #     sentences_fr.append(dest)
    # sentences_eng = "\n".join(sentence for sentence in sentences_eng)
    # sentences_fr = "\n".join(sentence for sentence in sentences_fr)
    #
    # with open(PROJECT_DIR / 'data/english_sentences_not_normalized.txt', 'w') as eng:
    #     eng.write(sentences_eng)
    # with open(PROJECT_DIR / 'data/french_sentences_not_normalized.txt', 'w') as fr:
    #     fr.write(sentences_fr)
    #
    # spm.SentencePieceTrainer.train(input=PROJECT_DIR / 'data/english_sentences_not_normalized.txt',
    #                                model_prefix=PROJECT_DIR / 'model/en_segmentator_not_normalized', vocab_size=1000,
    #                                pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    # spm.SentencePieceTrainer.train(input=PROJECT_DIR / 'data/french_sentences_not_normalized.txt',
    #                                model_prefix=PROJECT_DIR / 'model/fr_segmentator_not_normalized', vocab_size=1000,
    #                                pad_id=0, unk_id=1, bos_id=2, eos_id=3)

    s_eng = spm.SentencePieceProcessor(model_file=str(PROJECT_DIR / 'model/en_segmentator_not_normalized.model'))
    ids = s_eng.encode('New York', out_type=str)
    print(ids)


if __name__ == '__main__':
    main()
