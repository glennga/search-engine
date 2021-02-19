import argparse
import pickle
import time

# Required for pickle!
from index import IndexDescriptor


def _generator_file(in_fp):
    """ Deserialize a file into an iterator of the list of its pickled objects. """
    def _entry_generator():
        try:
            while True:
                entry = pickle.load(in_fp)
                yield entry

        except EOFError:
            return

    return _entry_generator()


def _search(in_fp, in_desc, word):
    """ Search for a given word using the index descriptor and the index file. We return the **1st** result. """
    designated_tell = in_desc[word]
    if designated_tell is None:
        print(f'Could not find larger entry in index, starting from position {0}')
        designated_tell = 0
    else:
        print(f'Starting from position {designated_tell}.')

    in_fp.seek(designated_tell)
    search_generator = _generator_file(in_fp)
    try:
        while True:
            token, postings_count = next(search_generator)
            if token == word:
                return next(search_generator)
            else:
                [next(search_generator) for _ in range(postings_count)]

    except EOFError:
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
            generator = _generator_file(index_fp)
            for entry in generator:
                print(entry)

        else:
            descriptor = next(_generator_file(desc_fp))
            before_search = time.process_time()
            print(_search(index_fp, descriptor, command_line_args.word))
            after_search = time.process_time()
            print(f"Search from index = {1000.0 * (after_search - before_search)}ms")
