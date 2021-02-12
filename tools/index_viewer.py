import argparse
import pickle


def _deserialize_generator(in_fp):
    """ Deserialize the given ordered disk component to the an ordered list of <token, inverted list> pairs.

    To avoid having to load the entire disk component into memory, we return a generator of <token, inverted list>
    pairs. The order in which the generator returns these pairs is the same order in which it was serialized (in
    alphabetical order of the tokens).

    :param in_fp: Binary file pointer of the disk component to deserialize.
    :return: A generator of <token, inverted list> pairs.
    """

    def _token_pair_generator():
        try:
            while True:
                yield pickle.load(in_fp)

        except EOFError:
            return

    return _token_pair_generator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View the contents of an index file, in sequential order.')
    parser.add_argument('index', type=str, help='Location of the index file to read.')
    command_line_args = parser.parse_args()

    with open(command_line_args.index, 'rb') as index_fp:
        generator = _deserialize_generator(index_fp)
        for entry in generator:
            print(entry)
