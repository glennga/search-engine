import argparse
import json


class Tokenizer:
    pass


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
        pass

    def index(self):
        for file in load(self.corpus):
            pass
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an inverted index given some corpus.')
    parser.add_argument('corpus', type=str, help='Location of the corpus to index.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Invoke our indexer.
    Indexer(command_line_args.corpus, **main_config_json).index()
