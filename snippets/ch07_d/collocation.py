import nltk.corpus
from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures


def rank_quadgrams(corpus, metric, path=None):
    """
    Find and rank quadgrams from the supplied corpus using the given
    association metric. Write the quadgrams out to the given path if
    supplied otherwise return the list in memory.
    """

    # Create a collocation ranking utility from corpus words.
    ngrams = QuadgramCollocationFinder.from_words(corpus.words())

    # Rank collocations by an association metric
    scored = ngrams.score_ngrams(metric)

    if path:
        with open(path, 'w', encoding="utf-8") as f:
            f.write("Collocation\tScore ({})\n".format(metric.__name__))
            for ngram, score in scored:
                f.write("{}\t{}\n".format(repr(ngram), score))
    else:
        return scored


if __name__ == '__main__':
    from reader import PickledCorpusReader
    corpus = PickledCorpusReader('../corpus.dict')
    rank_quadgrams(nltk.corpus.genesis, QuadgramAssocMeasures.likelihood_ratio, "quadgrams.txt")
