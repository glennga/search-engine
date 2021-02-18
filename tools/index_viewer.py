import argparse
import pickle
import zlib

# Required for pickle!
from index import IndexDescriptor


def _generator(in_fp):
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


def _search(in_fp, in_desc, word):
    """ Search for a given word using the index descriptor and the index file. """
    designated_tell = in_desc[word]
    if designated_tell is None:
        print(f'Could not find larger entry in index, starting from position {0}')
        designated_tell = 0
    else:
        print(f'Starting from position {designated_tell}.')

    in_fp.seek(designated_tell)
    search_generator = _generator(in_fp)
    for data_entry in search_generator:
        if data_entry[0] == word:
            return {data_entry[0]: [pickle.loads(zlib.decompress(e)) for e in data_entry[1]]}

        elif data_entry[0] > word:
            print('Word does not exist!')
            return None

    print('Word does not exist!')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View the contents of an index file, in sequential order.')
    parser.add_argument('index', type=str, help='Location of the index file to read.')
    parser.add_argument('desc', type=str, help='Location of the descriptor file to read.')
    parser.add_argument('--word', type=str, default=None, help='Word to search using the index.')
    command_line_args = parser.parse_args()

    with open(command_line_args.index, 'rb') as index_fp, open(command_line_args.desc, 'rb') as desc_fp:
        if command_line_args.word is None:
            generator = _generator(index_fp)
            for entry in generator:
                print({entry[0]: [pickle.loads(zlib.decompress(e)) for e in entry[1]]})
        else:
            descriptor = next(_generator(desc_fp))
            print(_search(index_fp, descriptor, command_line_args.word))
