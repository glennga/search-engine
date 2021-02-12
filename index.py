import __init__
import argparse
import json
import sys
import datetime
from lxml import etree, html
from pathlib import Path

logger = __init__.get_logger('Index')


class Token:
    token = "example"
    docID = "0"
    frequency = 0
    tags = []

    def __init__(self, token, docID, frequency, tags):
        self.token = token
        self.docID = docID
        self.frequency = frequency
        self.tags = tags


class Tokenizer:

    def __init__(self):
        pass

    def tokenize(self, file):
        # Set up variables and get doc ID from file name.
        print(file.name.rstrip(".json"))
        docID = file.name.rstrip(".json")
        tokens = []
        url = ""
        data = {}
        encoding = ""

        # Populate the initialized variables from JSON
        with file.open() as f:
            data = json.loads(f.read())
            url = data["url"]
            content = data["content"]
            encoding = data["encoding"]

        # TODO: Tokenize content
        print(data)
        input()
        return tokens



class StorageHandler:

    def write(self, token, entry):
        """

        :param token:
        :param entry: [docID, frequency, <tuple of tags>]
        :return:
        """
        pass

    pass


class Indexer:
    def __init__(self, corpus, **kwargs):
        self.corpus = corpus
        self.config = kwargs
        self.tokenizer = Tokenizer()
        self.storage_handler = StorageHandler()
        pass

    def index(self):
        corpus_path = Path(self.corpus)
        for subdomain_directory in corpus_path.iterdir():
            if not subdomain_directory.is_dir():
                continue
            print(subdomain_directory)
            for file in subdomain_directory.iterdir():
                if not file.is_file():
                    continue
                print(file.name)
                tokens = self.tokenizer.tokenize(file)
                for token in tokens:
                    self.storage_handler.write(token.token, [token.docID, token.frequency, token.tags])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an inverted index given some corpus.')
    parser.add_argument('--corpus', type=str, default='DEV', help='Path to the directory of the corpus to index.')
    parser.add_argument('--config', type=str, default='config/index.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Invoke our indexer.
    Indexer(command_line_args.corpus, **main_config_json).index()
