

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
    def index(self, path):
        for file in load(path):
            pass
    pass



if __name__ == '__main__':
    # TODO
    pass