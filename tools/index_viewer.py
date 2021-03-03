import argparse
import pickle
import time

# Required for pickle!
from index import IndexDescriptor, SkipList


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
            next(search_generator)  # Skip over the posting skip list.
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
    parser.add_argument('--url', type=str, default=None, help='URL to search for in some posting.')
    command_line_args = parser.parse_args()

    with open(command_line_args.index, 'rb') as index_fp, open(command_line_args.desc, 'rb') as desc_fp:
        m_before_load = time.process_time()
        m_descriptor = next(_generator_file(desc_fp))
        m_after_load = time.process_time()
        print(f"Load descriptor = {1000.0 * (m_after_load - m_before_load)}ms")

        if command_line_args.word is None:
            m_generator = _generator_file(index_fp)
            m_count, md_count = 0, 0
            try:
                while True:
                    m_token, m_postings_count = next(m_generator)
                    m_skip_list = next(m_generator)
                    print(f'Working entry: ({m_token}, {m_postings_count})')
                    print(f'Working tell: {index_fp.tell()}')
                    print(f'Skip list for the associated posting:\n{m_skip_list}')
                    for _ in range(m_postings_count):
                        next(m_generator)

                    if not m_token.isdigit():
                        md_count += 1
                    m_count += 1

            except StopIteration:
                print(f"Total number of tokens in index: {m_count}")
                print(f"Total number of non-numeric tokens in index: {md_count}")

        else:
            m_before_search = time.process_time()
            print(_search(index_fp, m_descriptor, command_line_args.word))
            m_after_search = time.process_time()
            print(f"Search from index = {1000.0 * (m_after_search - m_before_search)}ms")
