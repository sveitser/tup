#!/usr/bin/python

# Run onlineTATM2

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb


def main():
    """
    Loads and analyzes tweets
    """

    # The number of documents to analyze each iteration
    batchsize = 1000
    # The total number of documents in Wikipedia
    D = 1.0e6
    # The number of topics
    K = 50

    # How many documents to look at
    if (len(sys.argv) < 2):
        documentstoanalyze = int(D/batchsize)
    else:
        documentstoanalyze = int(sys.argv[1])

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)
    
    # Load documents
    tweets = file('./tweets_linebyline.txt').readlines()

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.9)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, documentstoanalyze):
        
        # Give some documents to online LDA
        start = iteration*batchsize
        end = (iteration+1)*batchsize - 1
        (gamma, bound) = olda.update_lambda(tweets[start:end])
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(tweets[start:end], olda._vocab)
        perwordbound = bound * len(tweets[start:end]) / (D * sum(map(sum, wordcts)))
        print '%d (%d - %d):  rho_t = %f,  held-out perplexity estimate = \t %f' % \
            (iteration, start, end, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt('data_lda/lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('data_lda/gamma-%d.dat' % iteration, gamma)
            

if __name__ == '__main__':
    main()
