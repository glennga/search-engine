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
import uuid
import hashlib
import base64

from lxml import etree, html
from pathlib import Path
from collections import deque
from sortedcontainers import SortedList
from nltk.stem import PorterStemmer

logger = __init__.get_logger('Index')


class Tokenizer:
    class Token:
        """ Data class for a single token within a document. """
        def __init__(self, token, frequency, document_pos, tags):
            self.token = token
            self.frequency = frequency
            self.document_pos = document_pos
            self.tags = tags

        def __str__(self):
            return self.token

    class MetaDataToken:
        """ Data class for a meta-token, which are handled before tokens themselves. """
        def __init__(self, string):
            self.text = string
            self.tag = "meta"

        def __str__(self):
            return self.text

    def __init__(self):
        self.porter = PorterStemmer()
        self.logger = __init__.get_logger('Tokenizer')

    def tokenize(self, file):
        """ :return A list of tokens, the URL, and the token hash associated with this document. """

        # Set up variables and get the path from file name.
        doc_path = file.name.rstrip(".json")
        tokens = {}  # Use dict for easy lookup, then convert to list before returning.

        # Populate the initialized variables from JSON.
        self.logger.info(f"Now opening file: {doc_path}.json")
        with file.open() as f:
            data = json.loads(f.read())
            doc_path = data["url"]
            encoding = data["encoding"]
            content = data["content"].encode()
        self.logger.info(f"DocID starting with {doc_path[:6]} contains URL {doc_path} with encoding {encoding}.")

        try:
            # Get the XML content from the string.
            root = html.fromstring(content)  # .decode(encoding=encoding, errors="ignore"))

            # Get metadata and prepare the relevant tokens.
            meta_names = root.xpath("//meta[@name]")
            meta_names_content = []
            for meta_name in meta_names:
                attribute = meta_name.attrib
                try:
                    if attribute["name"] in ["title", "description", "author", "keywords"]:
                        content = ""
                        if "content" in attribute:
                            content += str(attribute["content"]) + " "
                        if "value" in attribute:
                            content += str(attribute["value"]) + " "
                        meta_names_content.append(self.MetaDataToken(content))
                except Exception as e:
                    self.logger.error(f"tokenize: cannot retrieve metadata for attribute: {meta_name.attrib} "
                                      f"in docID starting with {doc_path[:6]}: {e}. Skipping this metadata entry.")
            if len(meta_names_content) > 0:
                self.logger.info(f"Found metadata for docID starting with "
                                 f"{doc_path[:6]}: {[content.text for content in meta_names_content]}")

            # Process the metadata tokens and XML contents.
            token_count = 0
            for tag_objects in [meta_names_content, root.iter()]:
                for element in tag_objects:
                    try:
                        # Is it a comment? Skip.
                        if element.tag is etree.Comment:
                            continue
                        # Is the tag a script or style? Skip.
                        if element.tag in ["style", "script"]:
                            continue
                        # Is the text empty? Skip.
                        if element.text is None:
                            continue
                        # Get each word in the tag.
                        for word in re.split(r"[^a-zA-Z0-9]+", element.text):
                            # Porter stemming (as suggested).
                            word = self.porter.stem(word.lower())
                            # Does the word even have content?
                            if len(word) <= 0:
                                # Log that it is blank".
                                continue
                            # Is the word not in the tokens list?
                            if word not in tokens:
                                tokens[word] = self.Token(word, 1, [token_count], [str(element.tag)])
                            # The word is in the tokens list.
                            else:
                                token = tokens[word]
                                token.frequency += 1
                                token.document_pos.append(token_count)
                                if element.tag not in token.tags:
                                    token.tags.append(str(element.tag))
                            token_count += 1
                    except UnicodeDecodeError as e:
                        self.logger.error(f"Tokenizer: UnicodeDecodeError: {e}. Skipping element in docID "
                                          f"starting with {doc_path[:6]}.")
        except etree.ParserError as e:
            self.logger.error(f"Tokenizer: etree.ParserError: {e}. Aborting scanning docID starting "
                              f"with {doc_path[:6]}.")
            return list(), doc_path, None

        tokens = list(tokens.values())
        self.logger.debug(f"Tokens in URL {doc_path}, docID starting with {doc_path[:6]}: {[token.token for token in tokens]}")
        return tokens, doc_path, base64.b64encode(hashlib.md5(pickle.dumps(tokens)).digest()).decode('ascii')


class Posting:
    """ Class with handles a posting entry with potential skip pointers.

    The width of a posting MUST be fixed from the first pass of the last merge, so a posting consists of two parts:
    1. The width of the posting (POSTING_WIDTH), which is fixed to 32 bytes (1 byte for each digit).
    2. The actual posting itself. To account for the fact that we must perform two passes for the last merge, we must
       also be able to accurately compute the size of the skip information prior to knowing the position of the posting
       to skip to- hence, we also fix this position to 32 bytes.

    """
    END_SKIP_LABEL_MARKER = '$'
    POSTING_WIDTH_WIDTH = 32
    TELL_WIDTH = 32

    @staticmethod
    def serialize_width(width):
        return width.to_bytes(Posting.POSTING_WIDTH_WIDTH, byteorder='big')

    @staticmethod
    def deserialize_width(in_fp):
        return int.from_bytes(in_fp.read(Posting.POSTING_WIDTH_WIDTH), 'big')

    @staticmethod
    def skip_pointer_width(label, tell, count_n):
        fixed_width_tell = str(tell).zfill(Posting.TELL_WIDTH)
        return len(f',"skip_label":"{label}","skip_tell":"{fixed_width_tell}","skip_count":{count_n}'.encode('utf-8'))

    def __init__(self, doc_id, url, frequency, tags, position_v, skip_label=None, skip_tell=None, skip_count=None):
        self.doc_id = doc_id
        self.url = url
        self.frequency = frequency
        self.tags = tags
        self.position_v = position_v
        self.skip_label = skip_label
        self.skip_count = skip_count
        if type(skip_tell) == str:
            self.skip_tell = int(skip_tell)
        else:
            self.skip_tell = skip_tell

    def __getitem__(self, index):
        return [self.doc_id, self.url, self.frequency, self.tags,
                self.position_v, self.skip_label, self.skip_tell, self.skip_count][index]

    def __iter__(self):
        yield 'doc_id', self.doc_id
        yield 'url', self.url
        yield 'frequency', self.frequency
        yield 'tags', self.tags
        yield 'position_v', self.position_v
        if self.skip_label is not None and self.skip_tell is not None:
            yield 'skip_label', self.skip_label
            yield 'skip_tell', str(self.skip_tell).zfill(self.TELL_WIDTH)
            yield 'skip_count', self.skip_count


class IndexDescriptor:
    """ Descriptor for a given index file.

    In addition to holding metadata about the index itself (time of creation, the name of the index, size of the
    corpus indexed, etc...), this also holds a skip list for our vocabulary. If the vocabulary is too large to fit into
    memory, a smaller skip probability in the index.json file will allow vocabulary lookup with a skip list whose
    terminal nodes hold positions to the smallest but closest vocabulary term. From here, an index accessor will scan
    leftward until they hit their desired word.

    """
    class SkipList:
        """ Skip list for vocabulary terms. """
        END_LABEL = '$'
        START_LABEL = '^'

        class _Node:
            def __init__(self, label, tell, level):
                self.label = label
                self.tell = tell
                self.level = level
                self.sibling = None
                self.child = None

            def __str__(self):
                return f'<{self.label}, {self.tell}> @ {self.level}'

            def set_sibling(self, sibling):
                self.sibling = sibling

            def set_child(self, child):
                self.child = child

            def look_right(self):
                return self.sibling

            def look_down(self):
                return self.child

        def __init__(self, l1_label_tells, **kwargs):
            self.skip_probability = kwargs['skip_probability']
            self.l1_probability = kwargs['l1_probability']
            self.max_skip_height = kwargs['max_skip_height']
            self.sentinel = self._build(l1_label_tells)

        def __getitem__(self, item):
            """ Given an item, search our skip list and return the tell associated with the closest node.

            1) We start from the top node of our sentinel.
            2) Drop down and move right until we reach a tail node OR that node's label is greater than our query item.
            3) Return the lowest and closest node's tell.

            :param item: Label to search for.
            :return The tell associated with smallest closest label in our list.
            """
            current_node = self.sentinel
            while current_node.look_down() is not None:
                current_node = current_node.look_down()
                while current_node.look_right().label != self.END_LABEL and current_node.look_right().label < item:
                    current_node = current_node.look_right()

            return current_node.look_right().tell

        def __str__(self):
            """ Print something pretty :-) """
            def _exhaust_sll(node):
                resultant = ''
                while node.look_right().label != self.END_LABEL:
                    node = node.look_right()
                    resultant += f'({node.label}, {node.tell}) -- '
                return resultant

            complete_string = ''
            current_left_node = self.sentinel
            while current_left_node.look_down() is not None:
                current_left_node = current_left_node.look_down()
                partial_string = self.START_LABEL + ' -- ' + _exhaust_sll(current_left_node) + self.END_LABEL
                complete_string += partial_string + f'\n|{"".join("-" for _ in range(len(partial_string) - 2))}|\n'

            return complete_string

        @staticmethod
        def _link_nodes(node_list):
            """ Construct a SLL given an ordered list of nodes. We return the first node in this sequence. """
            current_node = node_list[0]
            for node in node_list[1:]:
                current_node.set_sibling(node)
                current_node = node

            return node_list[0]

        def _build(self, l1_label_tells):
            """ Build our skip list.

            1) Build the L1 layer. This will consist of all of the labels supplied to us by the caller.
            2) Build the next layer. This involves a) building the next sentinel node, b) iterating to the end of the
               previous sentinel node's siblings and c) determining whether a node survives or not (by skipProbability).
               We ensure a node's survival by creating a parent of the current node and linking it with the next
               sentinel node.
            3) Repeat this for maxSkipHeight times. We do not include layers that have no content nodes or whose content
               has not changed with the previous layer.

            :param l1_label_tells: <label, byte location> pairs that represent the L1 layer of the skip list.
            :return A list of nodes, representing the starting nodes for each layer.
            """
            l1_sentinel = self._link_nodes([self._Node(self.START_LABEL, None, 1)] +
                                           [self._Node(t[0], t[1], 1) for t in l1_label_tells] +
                                           [self._Node(self.END_LABEL, None, 1)])

            current_node = l1_sentinel
            current_height = 1
            current_length = len(l1_label_tells) + 2
            for level in range(1, self.max_skip_height):
                level_sentinel = self._Node(self.START_LABEL, None, current_height + 1)
                level_sentinel.set_child(current_node)
                level_nodes = [level_sentinel]

                while current_node.look_right() is not None:
                    if current_node.label != self.START_LABEL and random.random() < self.skip_probability:
                        new_node = self._Node(current_node.label, current_node.tell, current_height + 1)
                        new_node.set_child(current_node)
                        level_nodes.append(new_node)
                    current_node = current_node.look_right()
                current_tail = current_node

                if (len(level_nodes) == 1 and current_length == 3) or level == self.max_skip_height - 1:
                    level_tail = self._Node(self.END_LABEL, None, current_height + 1)
                    level_tail.set_child(current_tail)
                    current_node = self._link_nodes([level_sentinel, level_tail])
                    return current_node
                elif 2 <= len(level_nodes) < current_length - 1:
                    level_tail = self._Node(self.END_LABEL, None, current_height + 1)
                    level_tail.set_child(current_tail)
                    current_node = self._link_nodes(level_nodes + [level_tail])
                    current_height = current_height + 1
                    current_length = len(level_nodes) + 1
                else:
                    current_node = level_sentinel.look_down()

    def __init__(self, l1_token_tells, **kwargs):
        self.skip_probability = kwargs['storage']['vocab']['skipProbability']
        self.l1_probability = kwargs['storage']['vocab']['l1Probability']
        self.max_skip_height = kwargs['storage']['vocab']['maxSkipHeight']
        self.corpus_size = kwargs['corpusSize']
        self.index_name = kwargs['indexFile']
        self.corpus = kwargs['corpus']

        self.time_built = str(datetime.datetime.now().isoformat())
        self.sentinel = self.SkipList(l1_token_tells, skip_probability=self.skip_probability,
                                      l1_probability=self.l1_probability, max_skip_height=self.max_skip_height)

    def __getitem__(self, item):
        return self.sentinel[item]

    def get_metadata(self):
        """ Return everything about the index (except the skip list itself). """
        return {
            'indexFile': self.index_name,
            'timeBuilt': self.time_built,
            'corpus': self.corpus,
            'corpusSize': self.corpus_size,
            'skipListMeta': {
                'skipProbability': self.skip_probability,
                'l1Probability': self.l1_probability,
                'maxSkipHeight': self.max_skip_height
            }
        }


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
       file, while the latter gives metadata + a skip list for the data file itself.
    d) To "index" the index, we use a skip-list instead of building a more balanced tree. This is because building a
       tree requires another pass of our entire index (or some more complex bookkeeping of our index while we build
       it)- but building a skip list can be done after the merge process simply and efficiently by just sampling while
       we build the index.
    e) To assist searching at runtime, skip pointers are provided for each index entry (whose skip probability is
       tunable in index.json). This requires two passes at the last merge step: one to determine the positions of the
       places to skip too, and another to actually write the final postings to disk.
    f) Duplicate entries (i.e. having the same content) are removed on the final pass. Building the duplicate detector
       is handled outside of the storage handler.

    At a high level, the index can be thought of as a skip list, where each L0 node contains a token and a singly-linked
    list (w/ skip pointers) of postings.

    """
    class _DoubleComponent:
        """ Simulates a "reset" operation by just keeping another copy of the iterator we can jump back to. """
        def __init__(self, component_1, component_2):
            self.component_1 = component_1
            self.component_2 = component_2

    def __init__(self, corpus, postings_f, alias_f, **kwargs):
        # Ensure that the directory exists first...
        if not os.path.exists(kwargs['storage']['spillDirectory']):
            logger.info(kwargs['storage']['spillDirectory'] + ' not found. Building now.')
            os.makedirs(kwargs['storage']['spillDirectory'])
        if not os.path.exists(kwargs['storage']['dataDirectory']):
            logger.info(kwargs['storage']['dataDirectory'] + ' not found. Building now.')
            os.makedirs(kwargs['storage']['dataDirectory'])

        self.merge_queue = deque()
        self.memory_component = dict()
        self.postings_f = postings_f
        self.alias_f = alias_f
        self.config = kwargs
        self.config['corpus'] = corpus

    def write(self, token, entry):
        """ Store an index entry with respect to a single document.

        To account for large lists in memory, we use a SortedList structure which performs insertions in approximately
        O(log(n)) time. If the in-memory component is larger than our spill threshold, we spill to disk.

        :param token: The token to associate with the given entry.
        :param entry: The document descriptor, which is an arbitrary sequence of bytes.
        """
        if token not in self.memory_component:
            self.memory_component[token] = SortedList(key=self.postings_f)
        self.memory_component[token].add(entry)

        if sys.getsizeof(self.memory_component) > self.config['storage']['spillThreshold']:
            self._spill(False)

    def close(self, document_count) -> tuple:
        """ Merge all of our partial disk components and the current in-memory component to a single inverted file.

        1. If we do not have any disk components, then our entire index resides in memory. Spill this to disk, and move
           the result to the data directory.
        2. Otherwise, we need to perform some merging. If the in-memory component is currently populated, then merge the
           in-memory component and the first file in our merge queue. Add the resultant file to the queue.
        3. Pull two files from the merge queue and merge them together. Repeat this until we only have one file in our
           merge queue.
        4. Move the remaining disk component file to the specified data directory. The merge is finished.
        5. Write our metadata to a descriptor file. This includes the skip list for searching the index file.

        :param document_count: All documents that were tokenized (used in index descriptor).
        :return The names of the generated index file and descriptor file.
        """
        if len(self.merge_queue) == 0:
            logger.info('No disk components found. Writing in-memory component to disk.')
            l1_labels = self._spill(True)

        while len(self.merge_queue) > 1 or len(self.memory_component) != 0:
            merge_level = min(int(re.search(r'.*/d([0-9]).*', f).group(1)) for f in self.merge_queue) + 1
            run_file = f'd{merge_level}_' + str(uuid.uuid4()) + '.comp'
            logger.info(f'Starting merge to generate file {run_file}.')

            if len(self.memory_component) > 0:
                right_filename = self.merge_queue.popleft()
                with open(right_filename, 'rb') as right_fp_1, open(right_filename, 'rb') as right_fp_2, \
                     open(self.config['storage']['spillDirectory'] + '/' + run_file, 'wb') as out_fp:
                    logger.info(f'Performing merge on in-memory component and {right_fp_1} component.')
                    left_component = self._DoubleComponent(
                        self._iterator_dict(self.memory_component, lambda k: k[0]),
                        self._iterator_dict(self.memory_component, lambda k: k[0])
                    )
                    right_component = self._DoubleComponent(
                        self._iterator_pickle(right_fp_1),
                        self._iterator_pickle(right_fp_2)
                    )
                    l1_labels = self._merge(left_component, right_component, len(self.merge_queue) == 0, out_fp)
                    self.memory_component.clear()
                os.remove(right_filename)

            else:
                left_filename, right_filename = self.merge_queue.popleft(), self.merge_queue.popleft()
                with open(left_filename, 'rb') as left_fp_1, open(left_filename, 'rb') as left_fp_2, \
                     open(right_filename, 'rb') as right_fp_1, open(right_filename, 'rb') as right_fp_2, \
                     open(self.config['storage']['spillDirectory'] + '/' + run_file, 'wb') as out_fp:
                    logger.info(f'Performing merge on {left_fp_1} component and {right_fp_1} component.')
                    left_component = self._DoubleComponent(
                        self._iterator_pickle(left_fp_1),
                        self._iterator_pickle(left_fp_2)
                    )
                    right_component = self._DoubleComponent(
                        self._iterator_pickle(right_fp_1),
                        self._iterator_pickle(right_fp_2)
                    )
                    l1_labels = self._merge(left_component, right_component, len(self.merge_queue) == 0, out_fp)
                os.remove(left_filename)
                os.remove(right_filename)

            self.merge_queue.append(self.config['storage']['spillDirectory'] + '/' + run_file)

        data_file = os.path.basename(self.config['corpus']) + '.idx'
        os.rename(self.merge_queue.popleft(), self.config['storage']['dataDirectory'] + '/' + data_file)
        logger.info(f'Merge has finished. Index {data_file} has been built.')
        self.config['indexFile'] = data_file
        self.config['corpusSize'] = document_count

        descriptor_file = re.search(r'(.*).idx', data_file).group(1) + '.desc'
        with open(self.config['storage']['dataDirectory'] + '/' + descriptor_file, 'wb') as descriptor_fp:
            logger.info(f'Writing descriptor file {descriptor_file}.')
            pickle.dump(IndexDescriptor(l1_labels, **self.config), descriptor_fp)

        return data_file, descriptor_file

    def _spill(self, is_last_spill):
        """ Spill the current in-memory component to disk.

        1. Give total order to memory component. We will store each <token, inverted list> in alphabetical token order.
        2. Write each <token, inverted list> pair incrementally to the given file pointer. We are using "pickle" to
           serialize each pair.
        3. Randomly sample tokens to build the L1 layer of a skip list (only applicable if this is the last spill).
        4. If this is last spill (i.e. we have no disk components), serialize and write our postings with skip pointers.

        :param is_last_spill: Denotes whether or not to sample tokens for the skip list.
        :return A randomly sampled ordered list of <tokens, byte location> pairs.
        """
        spill_file = 'd0_' + str(uuid.uuid4()) + '.comp'
        logger.info(f'Spilling component {spill_file} to disk.')
        l1_labels = list()

        with open(self.config['storage']['spillDirectory'] + '/' + spill_file, 'wb') as out_fp:
            ordered_memory_component = sorted(self.memory_component.items(), key=self.postings_f)
            for token, postings in ordered_memory_component:
                if is_last_spill and random.random() < self.config['storage']['vocab']['l1Probability']:
                    logger.debug(f'Adding {(token, out_fp.tell(),)} to L1 layer of vocab skip list.')
                    l1_labels.append((token, out_fp.tell(),))

                logger.debug(f'Writing token {token} with {len(postings)} postings to disk.')
                if not is_last_spill:
                    pickle.dump((token, len(postings)), out_fp)
                    for raw_posting in postings:
                        pickle.dump(raw_posting, out_fp)
                else:
                    self._post(token, postings, postings, out_fp)

        self.memory_component.clear()
        self.merge_queue.append(self.config['storage']['spillDirectory'] + '/' + spill_file)
        return l1_labels

    def _merge(self, left_component, right_component, is_last_merge, out_fp):
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
        5. On the last pass, we serialize and write our postings to disk with skip pointers.

        :param left_component: Iterator that produces a sequence of token, then posting count, and then postings.
        :param right_component: Iterator that produces a sequence of token, then posting count, and then postings.
        :return A randomly sampled ordered list of <tokens, byte location> pairs.
        """
        def _advance(component):
            """ Return the token of the current cursor and the number of postings associated with the token. """
            try:
                token_1, posting_count_1 = next(component.component_1)
                token_2, posting_count_2 = next(component.component_2)
                if token_1 != token_2:
                    raise RuntimeError('Illegal state! Components are not aligned.')
                is_exhausted = False
            except StopIteration:
                token_1 = None
                posting_count_1 = 0
                is_exhausted = True
            return token_1, posting_count_1, is_exhausted

        def _postings(component, posting_count):
            """ Return the posting of the component at the current cursor for postings_count times. """
            for i in range(posting_count):
                yield next(component)

        def _write(token, posting_count, postings_1, postings_2):
            """ Write the inverted list to our file pointer. """
            if is_last_merge and random.random() < self.config['storage']['vocab']['l1Probability']:
                logger.debug(f'Adding {(token, out_fp.tell(), )} to L1 layer of vocab skip list.')
                l1_labels.append((token, out_fp.tell(), ))

            logger.debug(f'Writing token {token} with {posting_count} postings to disk.')
            if not is_last_merge:
                pickle.dump((token, posting_count), out_fp)
                for raw_posting in postings_1:
                    pickle.dump(raw_posting, out_fp)
                    next(postings_2)
            else:
                self._post(token, postings_1, postings_2, out_fp)

        l1_labels = list()
        left_token, left_posting_count, is_left_exhausted = _advance(left_component)
        right_token, right_posting_count, is_right_exhausted = _advance(right_component)

        while True:
            left_iterator_1 = _postings(left_component.component_1, left_posting_count)
            left_iterator_2 = _postings(left_component.component_2, left_posting_count)
            right_iterator_1 = _postings(right_component.component_1, right_posting_count)
            right_iterator_2 = _postings(right_component.component_2, right_posting_count)

            if not is_left_exhausted and not is_right_exhausted:
                if left_token == right_token:
                    merged_postings_1 = heapq.merge(left_iterator_1, right_iterator_1, key=self.postings_f)
                    merged_postings_2 = heapq.merge(left_iterator_2, right_iterator_2, key=self.postings_f)
                    _write(left_token, left_posting_count + right_posting_count, merged_postings_1, merged_postings_2)
                    left_token, left_posting_count, is_left_exhausted = _advance(left_component)
                    right_token, right_posting_count, is_right_exhausted = _advance(right_component)

                elif left_token > right_token:
                    _write(right_token, right_posting_count, right_iterator_1, right_iterator_2)
                    right_token, right_posting_count, is_right_exhausted = _advance(right_component)

                else:  # left_token < right_token
                    _write(left_token, left_posting_count, left_iterator_1, left_iterator_2)
                    left_token, left_posting_count, is_left_exhausted = _advance(left_component)

            elif is_left_exhausted and not is_right_exhausted:
                _write(right_token, right_posting_count, right_iterator_1, right_iterator_2)
                right_token, right_posting_count, is_right_exhausted = _advance(right_component)

            elif not is_left_exhausted and is_right_exhausted:
                _write(left_token, left_posting_count, left_iterator_1, left_iterator_2)
                left_token, left_posting_count, is_left_exhausted = _advance(left_component)

            else:  # is_left_exhausted and is_right_exhausted
                break

        return l1_labels

    def _post(self, token, postings_1, postings_2, out_fp):
        """ Dump our postings to disk and augment our postings with skip pointers.

        1. Perform our first pass. This entails determining which documents will act as anchors (i.e. hold skip pointer
           endpoints). We also perform duplicate elimination and URL aliasing at this step as well (for the sake of
           determining the proper endpoints).
        2. Perform the second pass. We must again perform duplicate elimination and URL aliasing, but this time we will
           write the postings to disk.

        The serialization of choice here is JSON. Initially we tried Pickle, but the serialization here is (for our
        intent and purpose) non-deterministic. This makes it impossible to accurately calculate the endpoints required
        for the skip pointers. In order to determine the size of the JSON before reading it, a fixed-width "width" block
        is reserved before each posting which holds the size we must parse before handing this off to the JSON
        deserializer.

        """
        def _serialize(_posting):
            return json.dumps(dict(_posting), separators=(',', ':')).encode('utf-8')

        logger.debug('Performing pass #1: Finding the documents and tells associated with the anchor slots.')
        visited_set, slot_set, previous_slot, slot_count = set(), set(), 0, 0
        anchor_queue, out_tell = deque(), out_fp.tell()
        for i, raw_posting in enumerate(postings_1):
            posting = Posting(*dict(raw_posting).values())
            posting.url = self.alias_f(raw_posting.url)
            if posting.url not in visited_set:
                slot_count += 1
                visited_set.add(posting.url)
                serialized_posting = _serialize(posting)
                if i != 0 and random.random() < self.config['storage']['posting']['skipProbability']:
                    slot_delta = slot_count - previous_slot - 1
                    out_tell += Posting.skip_pointer_width(self.postings_f(raw_posting), 0, slot_delta)
                    anchor_queue.append((out_tell, self.postings_f(raw_posting), slot_delta))
                    previous_slot = slot_count
                    slot_set.add(i)
                out_tell += Posting.POSTING_WIDTH_WIDTH + len(serialized_posting)
        out_tell += Posting.skip_pointer_width(Posting.END_SKIP_LABEL_MARKER, 0, slot_count - previous_slot)
        anchor_queue.append((out_tell, Posting.END_SKIP_LABEL_MARKER, slot_count - previous_slot))

        before_meta_tell = out_fp.tell()
        pickle.dump((token, len(visited_set)), out_fp)
        after_meta_tell = out_fp.tell()
        tell_to_add = after_meta_tell - before_meta_tell
        visited_set.clear()

        logger.debug('Performing pass #2: Write the final results to disk.')
        for i, raw_posting in enumerate(postings_2):
            raw_posting.url = self.alias_f(raw_posting.url)
            if raw_posting.url not in visited_set:
                visited_set.add(raw_posting.url)
                posting = Posting(*dict(raw_posting).values())
                if i == 0 or i in slot_set:
                    posting.skip_tell, posting.skip_label, posting.skip_count = anchor_queue.popleft()
                    posting.skip_tell += tell_to_add
                serialized_posting = _serialize(posting)
                out_fp.write(Posting.serialize_width(len(serialized_posting)))
                out_fp.write(serialized_posting)

        pass

    @staticmethod
    def _iterator_dict(in_dict, entry_f):
        """ Turn a given dictionary into an iterator that returns <key, count> and the consequent sorted entries.  """
        def _entry_generator():
            for k, v in sorted(in_dict.items(), key=entry_f):
                yield k, len(v)
                for p in v:
                    yield p

        return _entry_generator()

    @staticmethod
    def _iterator_pickle(in_fp):
        """ Deserialize a file into an iterator of the list of its pickled objects. """
        def _entry_generator():
            try:
                while True:
                    entry = pickle.load(in_fp)
                    logger.debug(f'Returning entry {entry} to caller.')
                    yield entry

            except EOFError:
                return

        return _entry_generator()

    @staticmethod
    def _iterator_json(in_fp):
        """ Deserialize a file into an iterator of Python dictionaries. """
        def _entry_generator():
            try:
                while True:
                    entry_width = Posting.deserialize_width(in_fp)
                    serialized_entry = in_fp.read(entry_width).decode('utf-8')
                    entry = json.loads(serialized_entry)
                    logger.debug(f'Returning entry {entry} to caller.')
                    yield entry

            except EOFError:
                return

        return _entry_generator()


class Indexer:
    """ The entry point for this program. """
    def __init__(self, corpus, **kwargs):
        self.config = kwargs
        self.alias_map = dict()

        self.tokenizer = Tokenizer()
        self.corpus_path = Path(corpus)
        self.postings_f = lambda e: e[0]  # We are going to sort by the document ID.
        # self.postings_f = lambda e: e[1]  # We are going to sort by term frequency.
        self.storage_handler = StorageHandler(str(self.corpus_path.absolute()), self.postings_f, self.alias, **kwargs)

    def alias(self, doc_id):
        return self.alias_map[doc_id]

    def index(self):
        logger.info(f"Corpus Path: {self.corpus_path.absolute()}.")
        document_count = 0
        token_dict = dict()

        for subdomain_directory in self.corpus_path.iterdir():
            if not subdomain_directory.is_dir():
                continue
            logger.info(f"Reading files in the subdomain directory {subdomain_directory}.")

            for file in subdomain_directory.iterdir():
                if not file.is_file():
                    continue

                logger.info(f"Tokenizing file {file.name}, in path {'/'.join(file.parts[1:])}.")
                tokens, url, token_hash = self.tokenizer.tokenize(file)
                if token_hash not in token_dict:
                    token_dict[token_hash] = [url]
                else:
                    token_dict[token_hash].append(url)
                document_length = sum(t.frequency for t in tokens)
                document_count += 1

                document_id = str(uuid.uuid4()).replace('-', '')
                if len(document_id) != 32:
                    raise RuntimeError("Document ID is not a fixed width!")

                for token in tokens:
                    entry = Posting(document_id, url, token.frequency / document_length, token.tags[:100],
                                    token.document_pos, )
                    logger.debug(f"Now passing token to storage handler: {token.token}: {entry}")
                    self.storage_handler.write(token.token, entry)

                # Break for now as a test.
                # self.storage_handler.close(document_count)
                # return

        # Close the storage handler. We construct a map of aliases to aid in duplicate elimination.
        representatives = [(min(u, key=lambda a: len(a)), u) for u in token_dict.values()]
        for representative, urls in representatives:
            for url in urls:
                if url not in self.alias_map:
                    self.alias_map[url] = representative
                elif url in self.alias_map and self.alias_map[url] != representative:
                    raise RuntimeError('Illegal state: Stored alias is different than the working alias.')
        self.storage_handler.close(document_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an inverted index given some corpus.')
    parser.add_argument('--corpus', type=str, default='DEV', help='Path to the directory of the corpus to index.')
    parser.add_argument('--config', type=str, default='config/index.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Invoke our indexer.
    Indexer(command_line_args.corpus, **main_config_json).index()
