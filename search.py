import __init__
import argparse
import json
import sys
import heapq
import math
import time
import pickle
import itertools
import operator

# Required for pickle!
from index import IndexDescriptor
from urllib.parse import urlparse
from nltk.stem import PorterStemmer
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QMainWindow, QVBoxLayout, QListWidget, \
    QWidget, QGridLayout, QListWidgetItem, QLabel


logger = __init__.get_logger('Search')


class Ranker:
    """ Given a collection of postings, return an iterator that returns URLs in ranked order. """
    class RankingHandler:
        """ Mainly for debugging purposes. Switch to the value based method when not debugging. """
        def __init__(self, is_v):
            self.rankings = dict()
            self.is_v = is_v

        def reset(self):
            self.rankings.clear()

        def pre_augment(self, url):
            if url not in self.rankings:
                self.rankings[url] = [0, 0, 0, 0] if not self.is_v else 0

        def augment(self, url, i, v):
            if self.is_v:
                self.rankings[url] += v
            else:
                self.rankings[url][i] += v

        def replace(self, old_url, new_url):
            self.rankings[new_url] = self.rankings[old_url]
            del self.rankings[old_url]

        def __call__(self, *args, **kwargs):
            if self.is_v:
                return sorted(self.rankings.items(), key=lambda j: j[1], reverse=True)
            else:
                return sorted(self.rankings.items(), key=lambda j: sum(j[1]), reverse=True)

    def __init__(self, **kwargs):
        self.ranking_handler = self.RankingHandler(True)
        self.config = kwargs

    @staticmethod
    def _tf(frequency):
        """ Current formula: returns the token frequency. """
        return frequency

    def _idf(self, document_count):
        """ Current formula: returns the log of total docs / docs that the token is in. """
        return math.log10(self.config['corpus_size'] / document_count)

    def _tf_idf(self, frequency, document_count):
        """ Return the multiplication of the above two functions. """
        return self.config['ranker']['composite']['tfIdf'] * self._tf(frequency) * self._idf(document_count)

    def _tag_value(self, token_tags):
        """
        Current formula: Given a criteria of tags, assign a score if the word has been tagged at least once given
        the tag. These values aren't supported by science LOL
        """
        tag_score = 1
        for tag in token_tags:
            if tag in self.config['ranker']['tags']:
                tag_score += self.config['ranker']['tags'][tag]

        return self.config['ranker']['composite']['tags'] * tag_score

    def _depth_value(self, url):
        """ Current formula: the path length is inversely proportional to this value. """
        return self.config['ranker']['composite']['urlDepth'] * 1.0 / len(urlparse(url).path.split('/'))

    def _ngram_boost(self, combined_document_pos, search_length, tolerance):
        """ Current formula: If we find a contiguous sequence of {search_length} positions, increase rank. """
        if search_length == 1:
            return

        class GroupByKey:
            """
            For use with itertools.groupby. We basically create a new group when the difference between elements is
            greater than some threshold.
            """
            def __init__(self):
                self.prev = None
                self.flag = 1

            def __call__(self, *args, **kwargs):
                if self.prev and abs(self.prev - args[0][1][0]) > tolerance:
                    self.flag *= -1  # Create a new group!
                self.prev = args[0][1][0]
                return self.flag

        for url, document_pos in combined_document_pos.items():
            if len(document_pos) < search_length:
                continue

            for _, g in itertools.groupby(enumerate(document_pos), GroupByKey()):
                consecutive_grouping = list(map(operator.itemgetter(1), g))
                if len(set(c[1] for c in consecutive_grouping)) >= search_length:
                    self.ranking_handler.augment(url, 3, 1.0 * self.config["ranker"]["documentPos"]["weight"])

    def _bigram_boost(self, combined_document_pos):
        """ Current formula: if the tokens share document positions close to each other, increase rank. """
        for url, document_pos in combined_document_pos.items():
            if len(document_pos) <= 1:
                continue

            for i in range(len(document_pos) - 1):
                if abs(document_pos[i] - document_pos[i + 1]) <= 1:
                    # logger.debug(f"{url} has potential bigrams, increase rank.")
                    self.ranking_handler.augment(url, 3, 1.0 * self.config["ranker"]["documentPos"]["weight"])
                    break

    def rank(self, index_entries, pos_tolerance):
        """ Return a list of URLs in ranked order.

        1. Ensure that the search time is under some threshold (defined in our config file). This is accomplished by
           allocating search time to each query term (but allow unused time to spill over to other terms).
        2.

        :param index_entries: List of lists of token, document count, and iterator of postings.
                              [(token, document count, [posting 1, posting 2, ...] ), ...]
        :param pos_tolerance: Number of words stopped from the query, which specifies a tolerance for the ngram booster.
        :return: List of URLs in ranked order.
        """
        self.ranking_handler.reset()
        combined_document_pos = dict()  # {url: [combined_document_pos]}
        t_entry_prev = time.process_time()
        allocated_search_time = [
            t_entry_prev + max((self.config['ranker']['maximumSearchTime'] - 0.05) / len(index_entries),
                               self.config['ranker']['minimumTimePerEntry']) * i
            for i in range(1, len(index_entries) + 1)
        ]

        for k, entry in enumerate(index_entries):
            token, postings_count, postings = entry
            logger.debug(f'Now ranking entry ({token}, {postings_count}, {postings}).')
            processed_token_hashes = dict()   # {token_hash: url}

            for p in range(postings_count):
                if time.process_time() >= allocated_search_time[k] or p >= self.config['ranker']['maxPostings']:
                    break

                # Touch our disk! Get the posting.
                url, token_frequency, token_tags, token_document_pos, token_hash = next(postings)

                # Avoid duplicate documents.
                if token_hash in processed_token_hashes:
                    # logger.debug(f'{url} is similar to {processed_token_hashes[token_hash]}. Skipping ranking.')
                    if len(url) < len(processed_token_hashes[token_hash]):
                        # logger.debug(f'Current URL is smaller, so we are going to assume the URL {url}.')
                        combined_document_pos[url] = combined_document_pos[processed_token_hashes[token_hash]]
                        self.ranking_handler.replace(processed_token_hashes[token_hash], url)
                        del combined_document_pos[processed_token_hashes[token_hash]]

                        del processed_token_hashes[token_hash]
                        processed_token_hashes[token_hash] = url
                    continue
                else:
                    processed_token_hashes[token_hash] = url

                self.ranking_handler.pre_augment(url)
                self.ranking_handler.augment(url, 0, self._tf_idf(token_frequency, postings_count))
                self.ranking_handler.augment(url, 1, self._tag_value(token_tags))
                self.ranking_handler.augment(url, 2, self._depth_value(url))

                # Track the token document positions for later.
                new_document_pos = list((d, k) for d in token_document_pos)
                combined_document_pos[url] = new_document_pos if url not in combined_document_pos else \
                    list(heapq.merge(combined_document_pos[url], new_document_pos, key=lambda a: a[0]))

            t_current = time.process_time()
            logger.info(f'Processed {p + 1} number of postings.')
            logger.info(f'Time to rank entry {token}: {1000.0 * (t_current - t_entry_prev)}ms.')
            t_entry_prev = t_current

        self._ngram_boost(combined_document_pos, len(index_entries), pos_tolerance)
        return self.ranking_handler()


class Retriever:
    """ Retrieve a list of URLs for some query terms. This list should then be given to the Presenter. """
    def __init__(self, **kwargs):
        """ Open our descriptor and index files, and pass the corpus size to our ranker. """
        self.config = kwargs
        with open(self.config['desc_file'], 'rb') as descriptor_fp:
            t0 = time.process_time()
            self.descriptor = pickle.load(descriptor_fp)
            t1 = time.process_time()
            logger.info(f'Descriptor loaded. {self.descriptor.get_metadata()}')
            logger.info(f'Total time to load descriptor: {1000.0 * (t1 - t0)}ms')

        self.porter = PorterStemmer()
        self.ranker = Ranker(corpus_size=self.descriptor.get_metadata()['corpusSize'], **kwargs)

        # We perform stopping at the query layer.
        self.stop_words = set([
            self.porter.stem(w) for w in
            'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,'
            'but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,'
            'have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,'
            'may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,'
            'said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,'
            'this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,'
            'with,would,yet,you,your'.split(',')
        ])

    def __call__(self, *args, **kwargs):
        """ Given a list of search terms in args, return a list of URLs.

        1) For each search term, perform porter stemming. This is the same stemming used for our index.
        2) Avoid duplicates. Keep track of the terms we currently search for in the word set.
        3) Search our descriptor (skip list) for the now stemmed word.
        4) Move to the designated position in our index file.
        5) Begin a rightward search until we find our stemmed word. If the token we are currently on is less than
           our stemmed word, we continue the search. Otherwise, the word does not exist in our index.
        6) For each word that we find, add the (token, postings count, postings iterator) to a list we will give to
           our ranker.
        7) Feed this list to our ranker, and return its results.
        """
        logger.info(f'Retrieval has been invoked.')
        ranking_input = list()
        word_set = set()

        t0 = time.process_time()  # Perform stopping.
        original_search_terms = [self.porter.stem(w.lower()) for w in args]
        if not all(w in self.stop_words for w in original_search_terms):
            stopped_search_terms = [w for w in original_search_terms if w not in self.stop_words]
        else:
            stopped_search_terms = original_search_terms

        for search_term in stopped_search_terms:
            if search_term in word_set:
                continue
            else:
                word_set.add(search_term)

            designated_tell = self.descriptor[search_term]
            if designated_tell is None:
                logger.info(f'Could not find larger entry in index, starting from position {0}.')
                designated_tell = 0
            else:
                logger.info(f'Starting from position {designated_tell}.')

            index_fp = open(self.config['idx_file'], 'rb')
            index_fp.seek(designated_tell)
            search_generator = self._generator(index_fp)
            try:
                while True:
                    token, postings_count = next(search_generator)
                    if token == search_term:
                        logger.info(f'Word {search_term} found!')
                        ranking_input.append((token, postings_count, search_generator))
                        break
                    elif token < search_term:
                        logger.debug(f'Searching... Skipping over word {token}.')
                        [next(search_generator) for _ in range(postings_count)]
                    else:
                        logger.info(f'Word {search_term} not found. Not including in ranking.')
                        break

            except EOFError:
                logger.info(f'Word {search_term} not found [EOF reached]. Not including in ranking.')

        t1 = time.process_time()
        logger.info(f'Time to search words in skip list: {1000.0 * (t1 - t0)}ms.')
        ranking_output = self.ranker.rank(ranking_input, len(original_search_terms) - len(stopped_search_terms))

        t2 = time.process_time()
        logger.info(f'Time to fetch results + perform ranking: {1000.0 * (t2 - t1)}ms.')
        return ranking_output

    @staticmethod
    def _generator(in_fp):
        """ Deserialize a file into an iterator of the list of its pickled objects. """
        def _entry_generator():
            try:
                while True:
                    entry = pickle.load(in_fp)
                    yield entry

            except EOFError:
                return

        return _entry_generator()


class Presenter:
    """ User interface for the search engine. Given query terms, we should present a list of URLs to the user. """
    class Controller:
        def __init__(self, view, **kwargs):
            """ To complete the 'MVC' portion, the retriever acts as our model. """
            self.retriever = Retriever(**kwargs)
            self.config = kwargs
            self.view = view
            self._connect_signals()

            # State for our results list.
            self.working_results = list()
            self.results_cursor = 0

        def _connect_signals(self):
            """ Setup the signals associated with each button. """
            self.view.search_button.clicked.connect(lambda: self._search_action())
            self.view.prev_button.clicked.connect(lambda: self._prev_action())
            self.view.next_button.clicked.connect(lambda: self._next_action())
            self.view.search_button.setAutoDefault(True)
            self.view.prev_button.setAutoDefault(True)
            self.view.next_button.setAutoDefault(True)

        def _search_action(self):
            """ Fetch the text from our search bar and give this to our retriever. Add the results to our list. """
            search_text_terms = self.view.search_entry.text().split()
            logger.info(f'Searching with text terms: {search_text_terms}.')

            t0 = time.process_time()
            self.working_results = self.retriever(*search_text_terms)
            t_delta = 1000.0 * (time.process_time() - t0)
            logger.info(f'Searching + ranking finished in {t_delta} ms.')
            self.view.search_label.setText(f'{len(self.working_results)} results found in {str(t_delta)[:6]}ms!')

            self.results_cursor = 0
            self._display_results()

        def _display_results(self):
            """ Display the results, given our search cursor and results list."""
            upper_bound = min(self.results_cursor + self.config['presentation']['resultsPerPage'],
                              len(self.working_results))
            lower_bound = self.results_cursor
            logger.info(f'Displaying results from {lower_bound} to {upper_bound}.')

            self.view.prev_button.show()
            self.view.prev_spacer.hide()
            self.view.next_button.show()
            self.view.next_spacer.hide()
            if lower_bound == 0:
                self.view.prev_button.hide()
                self.view.prev_spacer.show()
            if upper_bound == len(self.working_results):
                self.view.next_button.hide()
                self.view.next_spacer.show()

            self.view.results_label.setText(f'{lower_bound + 1} to {upper_bound} results displayed.')
            self.view.results_list.clear()
            for i, url in enumerate(self.working_results[lower_bound:upper_bound]):
                logger.debug(f'Adding result URL {url[0]} of score(s) {url[1]} to display.')
                self.view.add_result(url[0], i + lower_bound + 1)

        def _prev_action(self):
            logger.info('"Previous Page" button clicked.')
            self.results_cursor = self.results_cursor - self.config['presentation']['resultsPerPage']
            self._display_results()

        def _next_action(self):
            logger.info('"Next Page" button clicked.')
            self.results_cursor = self.results_cursor + self.config['presentation']['resultsPerPage']
            self._display_results()

    class View(QMainWindow):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs

            # Setup our style.
            with open(self.config['style']) as css_fp:
                self.setStyleSheet(css_fp.read())

            # Setup our window.
            self.resize(*self.config['presentation']['startingWindowSize'])
            self.setWindowTitle('"Watch Out Google" Search Engine')

            # Setup our central widget.
            self.main_layout = QVBoxLayout()
            self.central_widget = QWidget(self)
            self.setCentralWidget(self.central_widget)
            self.central_widget.setLayout(self.main_layout)

            # Create the rest of the interface.
            self._build_search_bar()
            self._build_results_list()

        def add_result(self, url, i):
            list_label = QLabel(self)
            list_label.setOpenExternalLinks(True)
            list_label.setTextFormat(Qt.RichText)
            list_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            list_label.setAlignment(Qt.AlignHCenter)
            list_label.setStyleSheet(f"font-size: {self.config['presentation']['resultItemFontSize']}px;")
            list_label.setText(f"""
                <table><tr>
                    <td><i>{i})</i></td>
                    <td><a href="{url}">{url}</a></td>
                </tr></table>
            """)

            list_item = QListWidgetItem()
            list_size = QSize(1, self.config['presentation']['resultItemFontSize'] * 2)
            list_item.setSizeHint(list_size)
            self.results_list.addItem(list_item)
            self.results_list.setItemWidget(list_item, list_label)

        def _build_search_bar(self):
            self.search_button = QPushButton("Search")
            self.search_entry = QLineEdit()
            search_bar_layout = QGridLayout()
            search_bar_layout.addWidget(self.search_entry, 0, 0)
            search_bar_layout.addWidget(self.search_button, 0, 1)
            self.main_layout.addLayout(search_bar_layout)
            self.search_label = QLabel(self)
            self.search_label.setText(self.config['presentation']['startingText'])
            self.search_label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.search_label)

        def _build_results_list(self):
            self.results_label = QLabel(self)
            self.prev_button = QPushButton("Previous Page")
            self.prev_spacer = QWidget(self)
            self.next_button = QPushButton("Next Page")
            self.next_spacer = QWidget(self)
            results_navigation_layout = QGridLayout()
            results_navigation_layout.addWidget(self.prev_button, 0, 0, 1, 1)
            results_navigation_layout.addWidget(self.prev_spacer, 0, 0, 1, 1)
            results_navigation_layout.addWidget(self.results_label, 0, 1, 1, 7, Qt.AlignCenter)
            results_navigation_layout.addWidget(self.next_button, 0, 8, 1, 1)
            results_navigation_layout.addWidget(self.next_spacer, 0, 8, 1, 1)
            self.results_list = QListWidget(self)
            self.results_list.setItemAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.results_list)
            self.main_layout.addLayout(results_navigation_layout)

            # We hide the results buttons initially.
            self.prev_button.hide()
            self.next_button.hide()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the search engine.')
    parser.add_argument('--idx_file', type=str, default='out/data/corpus.idx', help='Path to the index file.')
    parser.add_argument('--desc_file', type=str, default='out/data/corpus.desc', help='Path to the descriptor file.')
    parser.add_argument('--style', type=str, default='config/style.css', help='Path to the stylesheet file.')
    parser.add_argument('--config', type=str, default='config/search.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Start our search engine.
    main_application = QApplication(list())
    main_view = Presenter.View(style=command_line_args.style, **main_config_json)
    main_view.show()
    Presenter.Controller(main_view, idx_file=command_line_args.idx_file,
                         desc_file=command_line_args.desc_file, **main_config_json)
    sys.exit(main_application.exec_())
