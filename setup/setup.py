import os
import ssl
import sys

import wget as wget

DIRECTORY = '../data/downloads/'


def progress_bar(current, total, width=80):
    progress_message = "Progress: %d%% [%d / %d] MB" % (current / total * 100, current / 1000000, total / 1000000)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download(file):
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    url = ""
    out_file = DIRECTORY

    if file == 0:
        url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        out_file += "GoogleNews-vectors-negative300.bin.gz"
        if os.path.isfile(out_file[:-3]):
            return
        print("Downloading Google News Word2Vec embedding - file size: 1.65GB")
        import gdown
        gdown.download(url, output=out_file)
        import gzip
        import shutil
        print("Unzipping Google News Word2Vec\n")
        with gzip.open(out_file, 'rb') as f_in:
            with open(out_file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(out_file)
        return

    elif file == 1:
        url = "https://raw.githubusercontent.com/jneidel/job-titles/master/job-titles.txt"
        out_file += "jobs.txt"

    elif file == 2:
        try:
            import en_core_web_lg
            return
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_lg")
            return

    elif file == 3:
        try:
            import en_core_web_sm
            return
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            return

    elif file == 4:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        return

    if os.path.isfile(out_file):
        return

    ssl._create_default_https_context = ssl._create_unverified_context
    wget.download(url, out=out_file, bar=progress_bar)


if __name__ == "__main__":
    download(0)
    download(1)
    download(2)
    download(3)
    download(4)
