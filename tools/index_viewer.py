import argparse
import pickle
import time
import json

# Required for pickle!
from index import IndexDescriptor, Posting


def _generator_pickle(in_fp):
    """ Deserialize a file into an iterator of the list of its pickled objects. """
    def _entry_generator():
        try:
            while True:
                entry = pickle.load(in_fp)
                yield entry
        except EOFError:
            return

    return _entry_generator()


def _generator_json(in_fp):
    """ Deserialize a file into an iterator of Python dictionaries. """
    def _entry_generator():
        try:
            while True:
                entry_width = Posting.deserialize_width(in_fp)
                serialized_entry = in_fp.read(entry_width).decode('utf-8')
                entry = json.loads(serialized_entry)
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
    search_generator = _generator_pickle(in_fp)
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


def _exhaust(in_fp):
    """ Iterate through all entries in our index. """
    search_generator = _generator_pickle(index_fp)
    token_count, n_token_count = 0, 0
    try:
        while True:
            previous_tell = index_fp.tell()
            token, postings_count = next(search_generator)
            after_tell = index_fp.tell()
            postings_generator = _generator_pickle(in_fp)
            print(f'[{previous_tell} - {after_tell}]: New entry! ({token}, {postings_count})')

            current_s = 0
            while current_s < postings_count:
                previous_tell = index_fp.tell()
                posting = Posting(*next(postings_generator).values())
                after_tell = index_fp.tell()
                print(f'[{previous_tell} - {after_tell}]: --- Working posting for URL {current_s}: {dict(posting)}.')
                if posting.skip_label is not None:
                    print(f'[{previous_tell} - {after_tell}]: ------ Has skip pointer to {posting.skip_label} with '
                          f'tell {posting.skip_tell} and skip count {posting.skip_count}.')
                    index_fp.seek(posting.skip_tell)
                    if posting.skip_label == Posting.END_SKIP_LABEL_MARKER:
                        break
                    else:
                        current_s += posting.skip_count

                else:
                    current_s += 1

            if not token.isdigit():
                n_token_count += 1
            token_count += 1

    except StopIteration or OverflowError:
        print(f'--------------------------------------')
        print(f"Total number of tokens in index: {token_count}")
        print(f"Total number of non-numeric tokens in index: {n_token_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View the contents of an index file, in sequential order.')
    parser.add_argument('index', type=str, help='Location of the index file to read.')
    parser.add_argument('desc', type=str, help='Location of the descriptor file to read.')
    parser.add_argument('--word', type=str, default=None, help='Word to search using the index.')
    command_line_args = parser.parse_args()

    with open(command_line_args.index, 'rb') as index_fp, open(command_line_args.desc, 'rb') as desc_fp:
        m_before_load = time.process_time()
        m_descriptor = next(_generator_pickle(desc_fp))
        m_after_load = time.process_time()
        print(f"Load descriptor = {1000.0 * (m_after_load - m_before_load)}ms")

        if command_line_args.word is None:
            m_before_search = time.process_time()
            _exhaust(index_fp)
            m_after_search = time.process_time()
            print(f"Iterating over index = {1000.0 * (m_after_search - m_before_search)}ms")

        else:
            m_before_search = time.process_time()
            _search(index_fp, m_descriptor, command_line_args.word)
            m_after_search = time.process_time()
            print(f"Search from index = {1000.0 * (m_after_search - m_before_search)}ms")
