Tuning document rankings is done by analyzing the following properties via black-box testing:
- tf-idf,
- tags that are labeled on the words themselves,
- the depth of the URL.

tf-idf is the most important ranking measurement for our search engine and thus has the highest weight for
our overall ranking of the documents.

We then take into account the tags that surround the term in the html file at least once in the document.
We considered all default html headers as important, with the first header being the most important.
Bold words are treated with the same importance as the lower headers, such as the fifth and sixth header.
We consider the title and meta tags as important, and will greatly boost the search results that have
the search query in the title and/or meta tags. This tuning is based off black box testing as well as
comparing the results to the current UCI search engine implemented by OIT and powered by SearchBlox.

We finally take into account the term positions in the document and boost the ones that see tokens
appearing early in the document.
