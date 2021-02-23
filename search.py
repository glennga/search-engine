import __init__
import argparse
import json
import sys
import math
import time
import pickle

# Required for pickle!
from index import IndexDescriptor
from nltk.stem import PorterStemmer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QMainWindow, QVBoxLayout,  QListWidget,\
    QWidget, QGridLayout, QListWidgetItem, QLabel


logger = __init__.get_logger('Search')


class Ranker:
    """ Given a collection of postings, return an iterator that returns URLs in ranked order. """
    # TODO: We should incorporate the following: tf-idf, "important words", anchor text, word positions.
    # TODO: We should remove duplicate documents based on the token hash.
    def __init__(self, **kwargs):
        self.config = kwargs

    def tf(self, frequency):
        # TODO: do we use a more sophisticated formula?
        # Current formula: returns the token frequency.
        return frequency

    def idf(self, document_count):
        # TODO: how do we get the entire corpus count to calculate this
        # Current formula: returns the log of total docs / docs that the token is in.
        return math.log10(self.config['corpus_size'] / document_count)

    def tf_idf(self, frequency, document_count):
        # TODO: return the multiplication of the above two functions
        return self.tf(frequency) * self.idf(document_count)

    def rank(self, index_entries):
        """ Return an iterator of URLs in ranked order.

        :param index_entries: List of lists of token, document count, and iterator of postings.
                              [(token, document count, [posting 1, posting 2, ...] ), ...]
        :return: Iterator of URLs in ranked order and total number of results.
        """
        rankings = dict()  # {url: tf-idf}
        for entry in index_entries:
            token, postings_count, postings = entry
            logger.debug(f'Given entry ({token}, {postings_count}, {postings}).')

            for _ in range(postings_count):
                # entry = (url, token.frequency, token.tags, token.document_pos, token_hash,)
                url, token_frequency, token_tags, token_document_pos, token_hash = next(postings)
                rankings[url] = self.tf_idf(token_frequency, postings_count)

        return iter([x[0] for x in sorted(rankings.items(), key=lambda i: i[1])]), len(rankings)


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

    def __call__(self, *args, **kwargs):
        """ Given a list of search terms in args, return a list of URLs.

        1) For each search term, perform porter stemming. This is the same stemming used for our index.
        2) Search our descriptor (skip list) for the now stemmed word.
        3) Move to the designated position in our index file.
        4) Begin a rightward search until we find our stemmed word. If the token we are currently on is less than
           our stemmed word, we continue the search. Otherwise, the word does not exist in our index.
        5) For each word that we find, add the (token, postings count, postings iterator) to a list we will give to
           our ranker.
        6) Feed this list to our ranker, and return its results.
        """
        logger.info(f'Retrieval has been invoked.')
        ranking_input = list()

        for search_term in args:
            normalized_word = self.porter.stem(search_term.lower())
            designated_tell = self.descriptor[normalized_word]
            if designated_tell is None:
                logger.info(f'Could not find larger entry in index, starting from position {0}')
                designated_tell = 0
            else:
                logger.info(f'Starting from position {designated_tell}.')

            index_fp = open(self.config['idx_file'], 'rb')
            index_fp.seek(designated_tell)
            search_generator = self._generator(index_fp)
            try:
                while True:
                    token, postings_count = next(search_generator)
                    if token == normalized_word:
                        logger.info(f'Word {normalized_word} ({search_term}) found!')
                        ranking_input.append((token, postings_count, search_generator))
                        break

                    elif token < normalized_word:
                        logger.debug(f'Searching... Skipping over word {token}.')
                        [next(search_generator) for _ in range(postings_count)]
                    else:
                        logger.info(f'Word {search_term} not found. Not including in ranking.')
                        break

            except EOFError:
                logger.info(f'Word {search_term} not found [EOF reached]. Not including in ranking.')

        return self.ranker.rank(ranking_input)

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
            self.view = view
            self._connect_signals()

        def _connect_signals(self):
            """ Setup the signals associated with each button. """
            self.view.search_button.clicked.connect(lambda: self._display_results())

        def _display_results(self):
            """ Fetch the text from our search bar and give this to our retriever. Add the results to our list. """
            search_text_terms = self.view.get_search_terms()
            logger.info(f'Searching with text terms: {search_text_terms}.')

            t0 = time.process_time()
            results_iterator, results_count = self.retriever(*search_text_terms)
            t1 = time.process_time()
            logger.info(f'Searching finished in {1000.0 * (t1 - t0)}ms.')

            self.view.set_result_description(f'{results_count} results found in {1000.0 * (t1 - t0)}ms.')
            self.view.clear_results()
            for url in results_iterator:
                self.view.add_result(url)

    class View(QMainWindow):
        # TODO: Add incremental results display (maybe a top 50, then "next" and "prev" buttons).
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs

            # Setup our window.
            self.setWindowTitle('"Watch Out Google" Search Engine')
            self.resize(1280, 480)

            # Setup our central widget.
            self.main_layout = QVBoxLayout()
            self.central_widget = QWidget(self)
            self.setCentralWidget(self.central_widget)
            self.central_widget.setLayout(self.main_layout)

            # Create the rest of the interface.
            self._build_search_bar()
            self._build_results_list()

        def get_search_terms(self):
            return self.search_entry.text().split()

        def add_result(self, url):
            self.results_list.addItem(url)

        def clear_results(self):
            self.results_list.clear()

        def set_result_description(self, text):
            self.results_description.setText(text)

        def _build_search_bar(self):
            self.search_button = QPushButton("Search")
            self.search_entry = QLineEdit()
            search_bar_layout = QGridLayout()
            search_bar_layout.addWidget(self.search_button, 0, 1)
            search_bar_layout.addWidget(self.search_entry, 0, 0)
            self.main_layout.addLayout(search_bar_layout)
            self.search_description = QLabel(self)
            self.search_description.setText("Type in your query and hit search!")
            self.search_description.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.search_description)

        def _build_results_list(self):
            self.results_description = QLabel(self)
            self.results_description.setAlignment(Qt.AlignRight)
            self.results_list = QListWidget(self)
            self.main_layout.addWidget(self.results_list)
            self.main_layout.addWidget(self.results_description)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the search engine.')
    parser.add_argument('--idx_file', type=str, default='out/data/corpus.idx', help='Path to the index file.')
    parser.add_argument('--desc_file', type=str, default='out/data/corpus.desc', help='Path to the descriptor file.')
    parser.add_argument('--config', type=str, default='config/search.json', help='Path to the config file.')
    command_line_args = parser.parse_args()
    with open(command_line_args.config) as config_file:
        main_config_json = json.load(config_file)

    # Start our search engine.
    main_application = QApplication(list())
    main_view = Presenter.View(**main_config_json)
    main_view.show()
    Presenter.Controller(main_view, idx_file=command_line_args.idx_file,
                         desc_file=command_line_args.desc_file, **main_config_json)
    sys.exit(main_application.exec_())
