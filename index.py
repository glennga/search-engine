import __init__
import argparse
import json
import sys
import datetime
import pickle
import os
import re
import heapq
import random

from lxml import etree, html
from pathlib import Path
from collections import deque
from sortedcontainers import SortedList

from nltk.stem import PorterStemmer

logger = __init__.get_logger('Index')


class Tokenizer:
    porter = PorterStemmer()

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

        def __str__(self):
            return self.token

    def __init__(self):
        pass

    def tokenize(self, file):
        # Set up variables and get doc ID from file name.
        print(file.name.rstrip(".json"))
        docID = file.name.rstrip(".json")
        tokens = {} # use dict for easy lookup, then convert to list before returning
        url = ""
        data = {}
        encoding = ""

        # Populate the initialized variables from JSON
        with file.open() as f:
            data = json.loads(f.read())
            url = data["url"]
            encoding = data["encoding"]
            content = data["content"].encode(encoding=encoding)

        # TODO: Tokenize content
        try:
            root = html.fromstring(content)
            for element in root.iter():
                try:
                    # TODO: do we need to get tokens from the metadata?
                    if element.text is None:
                        continue
                    # Get each word in the tag
                    for word in re.split(r"[^a-zA-Z0-9]+", element.text):
                        # Porter stemming (as suggested)
                        word = self.porter.stem(word.lower())
                        # Does the word even have content
                        if len(word) <= 0:
                            # log that it is blank"
                            continue
                        # Is the word not in the tokens list
                        if word not in tokens:
                            tokens[word] = self.Token(word, docID, 1, [element.tag])
                        # The word is in the tokens list
                        else:
                            token = tokens[word]
                            token.frequency += 1
                            if element.tag not in token.tags:
                                token.tags.append(element.tag)
                except UnicodeDecodeError as e:
                    print(e)
        except etree.ParserError as e:
            print(e)
            return list()

        tokens = list(tokens.values())
        return tokens


class IndexDescriptor:
    """ Descriptor for a given index file. Holds a skip list to give positions to "skip" to in the index file. """
    class _Node:
        def __init__(self, token, tell):
            self.token = token
            self.tell = tell
            self.sibling = None
            self.child = None

        def set_sibling(self, sibling):
            self.sibling = sibling

        def set_child(self, child):
            self.child = child

        def move_right(self):
            return self.sibling

        def move_down(self):
            return self.child

    def __init__(self, l1_token_tells, **kwargs):
        self.skip_probability = kwargs['storage']['skipProbability']
        self.time_built = str(datetime.datetime.now().isoformat())
        self.index_name = kwargs['indexFile']
        self.corpus = kwargs['corpus']
        self.sentinel = self._Node(0, 0)
        # self.sentinel = self._build(l1_token_tells)

    def get_metadata(self):
        """ Return everything about the index (except the skip list itself). """
        return {'indexFile': self.index_name, 'timeBuilt': self.time_built, 'corpus': self.corpus}


class StorageHandler:
    """ Class to manage the storage of our inverted index.

    The StorageHandler class is responsible for the following:
    1. Managing an in-memory inverted index, represented as a hash map (python dictionary).
    2. Spilling these in-memory components to a new disk component when we pass some "spillThreshold".
    3. Merging all partial components into a single inverted file when "close" is called.

    Some notable features about this inverted index are:
    a) Each component is *immutable*. This allows us to simplify reasoning about flushing, merging and building a search
       structure, but at the cost of not being able to update values. ElasticSearch similarly adopts this practice of
       having immutable components.
    b) The merging process happens "all-at-once" once "close" is called (rather than a "rolling merge" approach).
    c) The resulting inverted index consists of a data file and an descriptor file. The former acts as the inverted
       file, while the latter gives metadata + a search tree for the data file itself.
    d) A single inverted list MUST be able to fit into memory, but <token, inverted list> pairs can extend out of
       memory. TODO: See if we can lax this constraint later.

    """
    def __init__(self, corpus, **kwargs):
        self.merge_queue = deque()
        self.memory_component = dict()
        self.config = kwargs
        self.config['corpus'] = corpus

    def write(self, token, entry):
        """ Store an index entry with respect to a single document.

        To account for large lists in memory, we use a SortedList structure which performs insertions in approximately
        O(log(n)) time. If the in-memory component is larger than our spill threshold, we spill to disk.

        :param token: The token to associate with the given entry.
        :param entry: The document descriptor, consisting of the following: [docID, frequency, <tuple of tags>]
        """
        if token not in self.memory_component:
            self.memory_component[token] = SortedList(key=lambda ell: ell[0])
        self.memory_component[token].add(entry)

        if sys.getsizeof(self.memory_component) > self.config['storage']['spillThreshold']:
            self._spill()

    def close(self) -> tuple:
        """ Merge all of our partial disk components and the current in-memory component to a single inverted file.

        1. If we do not have any disk components, then our entire index resides in memory. Spill this to disk, and move
           the result to the data directory.
        2. Otherwise, we need to perform some merging. If the in-memory component is currently populated, then merge the
           in-memory component and the first file in our merge queue. Add the resultant file to the queue.
        3. Pull two files from the merge queue and merge them together. Repeat this until we only have one file in our
           merge queue.
        4. Move the remaining disk component file to the specified data directory. The merge is finished.
        5. Write our metadata to a descriptor file. This includes the skip list for searching the index file.

        :return The names of the generated index file and descriptor file.
        """
        if len(self.merge_queue) == 0:
            logger.info('No disk components found. Writing in-memory component to disk.')
            self._spill()

        while len(self.merge_queue) > 1:
            merge_level = min(int(re.search(r'.*/d([0-9]).*', f).group(1)) for f in self.merge_queue) + 1
            run_file = f'd{merge_level}_' + str(datetime.datetime.now().isoformat()) + '.comp'
            logger.info(f'Starting merge to generate file {run_file}.')

            if len(self.memory_component) > 0:
                with open(self.merge_queue.popleft(), 'rb') as right_fp, \
                     open(self.config['storage']['spillDirectory'] + '/' + run_file, 'wb') as out_fp:
                    logger.info(f'Performing merge on in-memory component and {right_fp} component.')
                    left_component = sorted(self.memory_component.items(), key=lambda token: token[0])
                    right_component = self._generator(right_fp)
                    l1_token_tells = self._merge(left_component, right_component, out_fp)
                    self.memory_component.clear()

            else:
                with open(self.merge_queue.popleft(), 'rb') as left_fp, \
                     open(self.merge_queue.popleft(), 'rb') as right_fp, \
                     open(self.config['storage']['spillDirectory'] + '/' + run_file, 'wb') as out_fp:
                    logger.info(f'Performing merge on {left_fp} component and {right_fp} component.')
                    left_component = self._generator(left_fp)
                    right_component = self._generator(right_fp)
                    l1_token_tells = self._merge(left_component, right_component, out_fp)

            self.merge_queue.append(self.config['storage']['spillDirectory'] + '/' + run_file)

        data_file = str(datetime.datetime.now().isoformat()) + '.idx'
        os.rename(self.merge_queue.popleft(), self.config['storage']['dataDirectory'] + '/' + data_file)
        logger.info(f'Merge has finished. Index {data_file} has been built.')
        self.config['indexFile'] = data_file

        descriptor_file = re.search(r'(.*).idx', data_file).group(1) + '.desc'
        with open(self.config['storage']['dataDirectory'] + '/' + descriptor_file, 'wb') as descriptor_fp:
            logger.info(f'Writing descriptor file {descriptor_file}.')
            pickle.dump(IndexDescriptor(l1_token_tells, **self.config), descriptor_fp)

        return data_file, descriptor_file

    def _spill(self):
        """ Spill the current in-memory component to disk.

        1. Give total order to memory component. We will store each <token, inverted list> in alphabetical token order.
        2. Write each <token, inverted list> pair incrementally to the given file pointer. We are using "pickle" to
           serialize each pair.
        3. Randomly sample tokens to build the L1 layer of a skip list (only applicable if this is the last spill).

        :return A randomly sampled ordered list of <tokens, byte location> pairs.
        """
        spill_file = 'd0_' + str(datetime.datetime.now().isoformat()) + '.comp'
        logger.info(f'Spilling component {spill_file} to disk.')
        l1_token_tells = list()

        with open(self.config['storage']['spillDirectory'] + '/' + spill_file, 'wb') as spill_fp:
            ordered_memory_component = sorted(self.memory_component.items(), key=lambda token: token[0])
            for disk_entry in ordered_memory_component:
                if random.random() < self.config['storage']['skipProbability']:
                    l1_token_tells.append((disk_entry[0], spill_fp.tell(), ))

                native_entry = (disk_entry[0], list(disk_entry[1]))
                logger.debug(f'Writing entry {native_entry} to disk.')
                pickle.dump(native_entry, spill_fp)

        self.memory_component.clear()
        self.merge_queue.append(self.config['storage']['spillDirectory'] + '/' + spill_file)
        return l1_token_tells

    def _merge(self, left_component, right_component, out_fp):
        """ Merge the two components together in a "merge-join" fashion, writing to the given file pointer.

        1. Open a new disk component, prefixed with "d{n}", where n = the current merge depth.
        2. Open the disk component, and get a generator to the token pairs. The in-memory component will henceforth
           be called the "left" component and the disk component will be called the "right" component.
        3. Maintain two "logical" cursors: a left cursor and right cursor. Begin the merging process.
           a) If the left cursor token equals the right cursor token, merge the two lists and write the result to disk.
              Both inverted lists are already sorted, so we can perform a similar "merge" approach at the inverted list
              granularity. Advance both cursors.
           b) If the left cursor token is greater than the right cursor token, then write the right token pair to disk.
              Advance only the right cursor.
           c) If the left cursor token is less than the right cursor token, then write the left token pair to disk.
              Advance only the left cursor.
           d) If we have exhausted any of the two lists, then write the remainder of the to-be-exhausted list to disk.
        4. While performing the merge, we randomly sample tokens to build the first layer of our skip list. The skip
           list grows with smaller 'skipProbability', and shrinks with larger 'skipProbability'.

        :param left_component: Iterator that produces sorted <token, inverted list> pairs.
        :param right_component: Iterator that produces sorted <token, inverted list> pairs.
        :return A randomly sampled ordered list of <tokens, byte location> pairs.
        """
        def _advance(component):
            try:
                token = next(component)
                is_exhausted = False
            except StopIteration:
                token = None
                is_exhausted = True
            return token, is_exhausted

        l1_token_tells = list()
        left_token_pair, is_left_exhausted = _advance(left_component)
        right_token_pair, is_right_exhausted = _advance(right_component)

        while True:
            if not is_left_exhausted and not is_right_exhausted:
                if left_token_pair[0] == right_token_pair[0]:
                    out_entry = (left_token_pair[0], list(heapq.merge(left_token_pair[1], right_token_pair[1])))
                    left_token_pair, is_left_exhausted = _advance(left_component)
                    right_token_pair, is_right_exhausted = _advance(right_component)

                elif left_token_pair[0] > right_token_pair[0]:
                    out_entry = right_token_pair
                    right_token_pair, is_right_exhausted = _advance(right_component)

                else:  # left_token_pair[0] < right_token_pair[0]
                    out_entry = left_token_pair
                    left_token_pair, is_left_exhausted = _advance(left_component)

            elif is_left_exhausted and not is_right_exhausted:
                out_entry = right_token_pair
                right_token_pair, is_right_exhausted = _advance(right_component)

            elif not is_left_exhausted and is_right_exhausted:
                out_entry = left_token_pair
                left_token_pair, is_left_exhausted = _advance(left_component)

            else:  # is_left_exhausted and is_right_exhausted
                break

            if random.random() < self.config['storage']['skipProbability']:
                logger.debug(f'Adding {(out_entry[0], out_fp.tell(), )} to L1 layer of skip list.')
                l1_token_tells.append((out_entry[0], out_fp.tell(), ))

            logger.debug(f'Writing entry {out_entry} to disk.')
            pickle.dump(out_entry, out_fp)

        return l1_token_tells

    @staticmethod
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
                    token_pair = pickle.load(in_fp)
                    logger.debug(f'Returning entry {token_pair} to caller.')
                    yield token_pair

            except EOFError:
                return

        return _token_pair_generator()


class Indexer:
    def __init__(self, corpus, **kwargs):
        self.corpus = corpus
        self.config = kwargs

        self.tokenizer = Tokenizer()
        self.storage_handler = StorageHandler(corpus, **kwargs)
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
                    #print(f"{token.token}: {token.docID}, {token.frequency}, {token.tags}")
                    self.storage_handler.write(token.token, [token.docID, token.frequency, token.tags])
                # Break for now as a test
                #return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an inverted index given some corpus.')
    parser.add_argument('--corpus', type=str, default='DEV', help='Path to the directory of the corpus to index.')
    parser.add_argument('--config', type=str, default='config/index.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Invoke our indexer.
    Indexer(command_line_args.corpus, **main_config_json).index()
