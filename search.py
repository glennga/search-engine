import argparse
import json
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QMainWindow, QVBoxLayout,  QListWidget,\
    QWidget, QGridLayout, QListWidgetItem, QLabel


class Ranker:
    """ Given a collection of postings, return an iterator that returns URLs in ranked order. """
    # TODO: We should incorporate the following: tf-idf, "important words", anchor text, word positions.
    # TODO: We should remove duplicate documents based on the token hash.
    def __init__(self, **kwargs):
        self.config = kwargs

    def rank(self, index_entries):
        """ Return an iterator of URLs in ranked order.

        :param index_entries: List of lists of token, document count, and iterator of postings.
                              [(token, document count, [posting 1, posting 2, ...] ), ...]
        :return: Iterator of URLs in ranked order.
        """


class Retriever:
    """ Retrieve a list of URLs for some query terms. This list should then be given to the Presenter. """
    def __init__(self, **kwargs):
        self.ranker = Ranker(**kwargs)
        self.config = kwargs

    def __call__(self, *args, **kwargs):
        """ Given a list of search terms in args, return a list of URLs. """
        pass


class Presenter(QMainWindow):
    """ User interface for the search engine. Given query terms, we should present a list of URLs to the user. """
    class Controller:
        def __init__(self, view, **kwargs):
            self.retriever = Retriever(**kwargs)
            self.view = view
            self._connect_signals()

        def _connect_signals(self):
            # self.view.search_button.clicked.connect()
            pass

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

        # Setup our central widget.
        self.setWindowTitle('"Watch Out Google" Search Engine')
        self.main_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.main_layout)

        # Create the rest of the interface.
        self._build_search_bar()
        self._build_results_list()

    def add_result(self, url):
        pass

    def clear_results(self):
        pass

    def set_result_description(self, text):
        pass

    def _build_search_bar(self):
        self.search_button = QPushButton("Search")
        self.search_entry = QLineEdit()
        search_bar_layout = QGridLayout()
        search_bar_layout.addWidget(self.search_button, 0, 0)
        search_bar_layout.addWidget(self.search_entry, 0, 1)
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
    main_view = Presenter(**main_config_json)
    main_view.show()
    Presenter.Controller(main_view, idx_file=command_line_args.idx_file,
                         desc_file=command_line_args.desc_file, **main_config_json)
    sys.exit(main_application.exec_())
