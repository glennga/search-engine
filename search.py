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
import re
import bisect

# Required for pickle!
from index import IndexDescriptor, Posting
from sortedcontainers import SortedList
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
            self.urls = SortedList()
            self.is_v = is_v

        def reset(self):
            self.rankings.clear()

        def pre_augment(self, url):
            if url not in self.rankings:
                self.rankings[url] = [0, 0, 0, 0] if not self.is_v else 0
                self.urls.add(url)

        def augment(self, url, i, v):
            if self.is_v:
                self.rankings[url] += v
            else:
                self.rankings[url][i] += v

        def remove(self, url):
            if url in self.rankings:
                del self.rankings[url]
                self.urls.remove(url)

        def replace(self, old_url, new_url):
            self.rankings[new_url] = self.rankings[old_url]
            self.remove(old_url)

        def contains(self, url):
            return url in self.rankings

        def next_largest(self, url):
            largest_index = bisect.bisect_right(self.urls, url)
            if largest_index == len(self.urls):
                return None
            else:
                return self.urls[largest_index]

        def __call__(self, *args, **kwargs):
            if self.is_v:
                return sorted(self.rankings.items(), key=lambda j: j[1], reverse=True)
            else:
                return sorted(self.rankings.items(), key=lambda j: sum(j[1]), reverse=True)

    def __init__(self, **kwargs):
        self.ESTIMATED_GRAM_BOOST_TIME_S = 0.05
        self.ranking_handler = self.RankingHandler(True)
        self.config = kwargs

        if self.config['ranker']['maxPostings'] < 0:
            self.config['ranker']['maxPostings'] = math.inf

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
                if abs(document_pos[i][0] - document_pos[i + 1][0]) <= 1:
                    # logger.debug(f"{url} has potential bigrams, increase rank.")
                    self.ranking_handler.augment(url, 3, 1.0 * self.config["ranker"]["documentPos"]["weight"])
                    break

    def disjunctive_rank(self, index_entries, pos_tolerance, time_budget):
        """ Return a list of URLs in ranked order. This is a *disjunctive* search.

        1. Ensure that the search time is under some threshold (defined in our config file). This is accomplished by
           allocating search time to each query term (but allow unused time to spill over to other terms).
        2. Iterate through each search term's posting list (while respecting the aforementioned time constraints).
        3. If the retrieved token hash associated with a posting matches that of another document found in that entry,
           ignore it. If this URL is smaller than the existing URL we have, replace it.
        4. Compute our weights and augment our rankings. These are tf-idf, tags (important words), and URL depth. Save
           our word positions until we finish processing all entries.
        5. Boost the ranks of URLs that have n-grams.
        6. Return our results.

        :param index_entries: List of lists of token, document count, skip list, and iterator of postings.
                              [(token, document count, [posting 1, posting 2, ...] ), ...]
        :param pos_tolerance: Number of words stopped from the query, which specifies a tolerance for the ngram booster.
        :param time_budget: Time left for the ranker.
        :return: List of URLs in ranked order.
        """
        self.ranking_handler.reset()
        combined_document_pos = dict()  # {url: [combined_document_pos]}
        t_entry_prev = time.process_time()
        allocated_search_time = [
            t_entry_prev + max((time_budget - self.ESTIMATED_GRAM_BOOST_TIME_S) / len(index_entries),
                               self.config['ranker']['minimumTimePerEntry']) * i
            for i in range(1, len(index_entries) + 1)
        ]

        for k, entry in enumerate(index_entries):
            token, postings_count, postings, _ = entry
            logger.debug(f'Now ranking entry ({token}, {postings_count}, {postings}).')

            for p in range(postings_count):
                if time.process_time() >= allocated_search_time[k] or p >= self.config['ranker']['maxPostings']:
                    break

                # Touch our disk! Get the posting.
                posting = Posting(*next(postings).values())

                self.ranking_handler.pre_augment(posting.url)
                self.ranking_handler.augment(posting.url, 0, self._tf_idf(posting.frequency, postings_count))
                self.ranking_handler.augment(posting.url, 1, self._tag_value(posting.tags))
                self.ranking_handler.augment(posting.url, 2, self._depth_value(posting.url))

                # Track the token document positions for later.
                new_document_pos = list((d, k) for d in posting.position_v)
                combined_document_pos[posting.url] = new_document_pos if posting.url not in combined_document_pos else \
                    list(heapq.merge(combined_document_pos[posting.url], new_document_pos, key=lambda a: a[0]))

            t_current = time.process_time()
            logger.info(f'Processed {p + 1} number of postings.')
            logger.info(f'Time to rank entry {token}: {1000.0 * (t_current - t_entry_prev)}ms.')
            t_entry_prev = t_current

        self._ngram_boost(combined_document_pos, len(index_entries), pos_tolerance)
        return self.ranking_handler()

    def conjunctive_rank(self, index_entries, pos_tolerance, time_budget):
        """ Return a list of URLs in ranked order. This is a *conjunctive* search.

        1. If only one term exists, defer this to the disjunctive_rank. Otherwise, proceed.
        2. We process the entries with the smallest number of postings first (think pessimistically first).
        3. Intersect the first two entries. We perform this intersection in O(m + n) time, where m and n refer to the
           number of postings in these two entries.
        4. Upon intersecting, a result will compute the appropriate weights and augment our rankings. These are tf-idf,
           tags (important words), and URL depth. Save our word positions until we finish processing all entries.
        5. Finish intersecting the remaining words.
        6. Remove duplicates (after the fact), by utilizing the token_hashes of all resultant URLs.
        7. Boost the ranks of URLs that have n-grams.
        8. Return our results.
        9. If at any point while intersecting, we run out of time, boost the n-grams and return the sorted list.

        :param index_entries: List of lists of token, document count, iterator of postings, and a skip function.
                              [(token, document count, [posting 1, posting 2, ...] ), ...]
        :param pos_tolerance: Number of words stopped from the query, which specifies a tolerance for the ngram booster.
        :param time_budget: Time left for the ranker.
        :return: List of URLs in ranked order.
        """
        if len(index_entries) <= 1:
            return self.disjunctive_rank(index_entries, pos_tolerance, time_budget)
        self.ranking_handler.reset()
        combined_document_pos = dict()

        # We process the entry with the least cardinality first.
        sorted_entries = sorted(index_entries, key=lambda i: i[1])
        token_left, count_left, postings_left, skip_left = sorted_entries[0]
        token_right, count_right, postings_right, skip_right = sorted_entries[1]

        # Perform the intersection for the first two entries.
        t_limit = time.process_time() + time_budget - self.ESTIMATED_GRAM_BOOST_TIME_S
        posting_left = Posting(*next(postings_left).values())
        posting_right = Posting(*next(postings_right).values())

        # Pointers for the intersection.
        consumed_left, consumed_right = 0, 0
        # last_left_skip, last_right_skip = None, None

        while consumed_left < count_left and consumed_right < count_right:
            if time.process_time() > t_limit:
                logger.info(f'Exiting early. Processed {consumed_left} postings of entry {token_left} and'
                            f' {consumed_right} postings of entry {token_right}.')
                self._ngram_boost(combined_document_pos, len(index_entries), pos_tolerance)
                return self.ranking_handler()

            # if posting_left.skip_label is not None:
            #     last_left_skip = (posting_left.skip_label, posting_left.skip_tell, posting_left.skip_count, )
            # if posting_right.skip_label is not None:
            #     last_right_skip = (posting_right.skip_label, posting_right.skip_tell, posting_right.skip_count, )

            if posting_left.url < posting_right.url:
                # if last_left_skip is not None and last_left_skip[0] <= posting_right.url and \
                #         last_left_skip[0] != Posting.END_SKIP_LABEL_MARKER:
                #     skip_left(last_left_skip[1])
                #     posting_left = Posting(*next(postings_left).values())
                #     consumed_left += last_left_skip[2]
                #     last_left_skip = None
                #
                # elif last_left_skip is not None and last_left_skip[0] < posting_right.url:
                #     skip_left(last_left_skip[1])
                #     consumed_left = count_left
                #
                # else:
                #     posting_left = Posting(*next(postings_left).values())
                #     consumed_left += 1
                consumed_left += 1
                if consumed_left < count_left:
                    posting_left = Posting(*next(postings_left).values())

            elif posting_right.url < posting_left.url:
                # if last_right_skip is not None and last_right_skip[0] <= posting_left.url and \
                #         last_right_skip[0] != Posting.END_SKIP_LABEL_MARKER:
                #     skip_right(last_right_skip[1])
                #     posting_right = Posting(*next(postings_right).values())
                #     consumed_right += last_right_skip[2]
                #     last_right_skip = None
                #
                # elif last_right_skip is not None and last_right_skip[0] < posting_left.url:
                #     skip_right(last_right_skip[1])
                #     consumed_right = count_right
                #
                # else:
                #     posting_right = Posting(*next(postings_right).values())
                #     consumed_right += 1
                consumed_right += 1
                if consumed_right < count_right:
                    posting_right = Posting(*next(postings_right).values())

            else:
                self.ranking_handler.pre_augment(posting_left.url)
                self.ranking_handler.augment(posting_left.url, 0, self._tf_idf(posting_left.frequency, count_left))
                self.ranking_handler.augment(posting_left.url, 0, self._tf_idf(posting_right.frequency, count_right))
                self.ranking_handler.augment(posting_left.url, 1, self._tag_value(posting_left.tags))
                self.ranking_handler.augment(posting_left.url, 1, self._tag_value(posting_right.tags))
                self.ranking_handler.augment(posting_left.url, 2, self._depth_value(posting_left.url))
                combined_document_pos[posting_left.url] = list(heapq.merge(
                    list((d, 0) for d in posting_left.position_v),
                    list((d, 1) for d in posting_right.position_v),
                    key=lambda a: a[0]))

                consumed_left += 1
                consumed_right += 1
                if consumed_left < count_left:
                    posting_left = Posting(*next(postings_left).values())
                if consumed_right < count_right:
                    posting_right = Posting(*next(postings_right).values())

        logger.info(f'Processed all postings of entries {token_left} and {token_right}.')

        # Intersect the remaining entries.
        for k, entry in enumerate(sorted_entries[2:]):
            token, count_n, postings, skip_f = entry
            keep_list = list()
            consumed = 0
            # last_skip = None

            while consumed < count_n:
                if time.process_time() > t_limit:
                    logger.info(f'Exiting early. Processed {consumed} postings of entry {token}.')
                    self._ngram_boost(combined_document_pos, len(index_entries), pos_tolerance)
                    return self.ranking_handler()

                posting = Posting(*next(postings).values())
                # if posting.skip_label is not None:
                #     last_skip = (posting.skip_label, posting.skip_tell, posting.skip_count, )

                if not self.ranking_handler.contains(posting.url):
                    # if last_skip is not None and last_skip[0] != Posting.END_SKIP_LABEL_MARKER:
                    #     next_largest = self.ranking_handler.next_largest(posting.url)
                    #     if next_largest is None:
                    #         break
                    #
                    #     elif next_largest < last_skip[0]:
                    #         skip_f(last_skip[1])
                    #         consumed += last_skip[2]
                    #         last_skip = None
                    #
                    #     else:
                    #         consumed += 1
                    #
                    # elif last_skip is not None:
                    #     consumed += 1
                    consumed += 1

                else:
                    keep_list.append(posting.url)
                    self.ranking_handler.augment(posting.url, 0, self._tf_idf(posting.frequency, count_n))
                    self.ranking_handler.augment(posting.url, 1, self._tag_value(posting.tags))
                    self.ranking_handler.augment(posting.url, 2, self._depth_value(posting.url))
                    combined_document_pos[posting.url] = list(heapq.merge(
                        combined_document_pos[posting.url],
                        list((d, k) for d in posting.position_v),
                        key=lambda a: a[0]))

            logger.info(f'Processed all postings of entry {token}.')
            for url in list(heapq.merge(keep_list, self.ranking_handler.urls)):
                self.ranking_handler.remove(url)

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
        if self.config['maximumSearchTime'] < 0:
            self.config['maximumSearchTime'] = math.inf

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
            entry_generator = self._generator_pickle(index_fp)
            try:
                while True:
                    token, postings_count = next(entry_generator)
                    # postings = self._generator_json(index_fp)
                    postings = self._generator_pickle(index_fp)

                    if token == search_term:
                        logger.info(f'Word {search_term} found!')
                        ranking_input.append((token, postings_count, postings, lambda a: index_fp.seek(a)))
                        break
                    elif token < search_term:
                        logger.debug(f'Searching... Skipping over word {token}.')
                        [next(entry_generator) for _ in range(postings_count)]
                    else:
                        logger.info(f'Word {search_term} not found. Not including in ranking.')
                        break

            except EOFError:
                logger.info(f'Word {search_term} not found [EOF reached]. Not including in ranking.')

        t1 = time.process_time()
        logger.info(f'Time to search words in skip list: {1000.0 * (t1 - t0)}ms.')
        ranking_output = self.ranker.conjunctive_rank(
            index_entries=ranking_input,
            pos_tolerance=len(original_search_terms) - len(stopped_search_terms),
            time_budget=self.config['maximumSearchTime'] - (t1 - t0),
        )

        t2 = time.process_time()
        logger.info(f'Time to fetch results + perform ranking: {1000.0 * (t2 - t1)}ms.')
        return ranking_output

    @staticmethod
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

    @staticmethod
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
            search_text_terms = re.split(r"[^a-zA-Z0-9]+", self.view.search_entry.text())
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

            self.view.results_label.setText(f'{(lower_bound + 1) if upper_bound != 0 else 0} '
                                            f'to {upper_bound} results displayed.')
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
            self.setWindowTitle('Omnigenix: Text is Knowledge')

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
