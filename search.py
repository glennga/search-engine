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
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QMainWindow, QVBoxLayout, QListWidget, \
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
        :return: List of URLs in ranked order.
        """
        rankings = dict()  # {url: tf-idf}
        for entry in index_entries:
            token, postings_count, postings = entry
            logger.debug(f'Given entry ({token}, {postings_count}, {postings}).')

            for _ in range(postings_count):
                # entry = (url, token.frequency, token.tags, token.document_pos, token_hash,)
                url, token_frequency, token_tags, token_document_pos, token_hash = next(postings)
                rankings[url] = self.tf_idf(token_frequency, postings_count)

        return [x[0] for x in sorted(rankings.items(), key=lambda i: i[1])]


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
            upper_bound = min(self.results_cursor + self.config['resultsPerPage'], len(self.working_results))
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
                logger.debug(f'Adding result URL {url} to display.')
                self.view.add_result(url, i + lower_bound + 1)

        def _prev_action(self):
            logger.info('"Previous Page" button clicked.')
            self.results_cursor = self.results_cursor - self.config['resultsPerPage']
            self._display_results()

        def _next_action(self):
            logger.info('"Next Page" button clicked.')
            self.results_cursor = self.results_cursor + self.config['resultsPerPage']
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
