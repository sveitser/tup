require("Distributions", "Winston")
using(Distributions)

function dirichlet_expecation(alpha)
    psigamma(alpha) - psigamma(sum(alpha))
end

# this should probably be a class
function online_lda(vocab, K, D, alpha, eta, tau0, kappa)

    lambda = rand(Gamma(100, 1 / 100), K, W)
    Elogbeta = dirichlet_expecation(lambda)
    expElogbeta = exp(Elogbeta)

    function parse_docs(docs)
        D = size(docs, 1)
        wordids = 
    end

    function do_e_step(docs::String)
        do_e_step([docs])
    end

    function do_e_step(docs)
        wordids, wordcounts = parse_doc_list(docs, vocab)
        batchD = size(docs, 1)
      
        gamma  = rand(Gamma(100, 1/100, batchD, K))
        Elogtheta = dirichlet_expecation(gamma)
        expElogtheta = exp(Elogbeta)

        sstats = zeros(size(lambda))
        meanchange = 0
        for d in 1:batchD
            phinorm = expElogtheta[d, :] * expElogbeta[d, :]' + 1e-100
            for it in 1:100
                lastgamma = gamma[d, :]
                gamma[d, :] = alpha + expElogbeta[d, :] * wordcounts[d] / phinorm  *
                    expElogbeta[d, :]
                    exp
        end
    end

    function update_lambda(docs)

    end

    function approx_bound(docs, gamma)

    end
end
