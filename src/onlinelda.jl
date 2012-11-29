require("Distributions", "Winston")
using(Distributions)

function dirichlet_expecation(alpha)
    psigamma(alpha) - psigamma(sum(alpha))
end

function online_lda(vocab, K, D, alpha, eta, tau0, kappa)

    lambda = rand(Gamma(100, 1 / 100), K, W)
    Elogbeta = dirichlet_expecation(lambda)
    expElogbeta = exp(Elogbeta)

    function do_e_step(docs)
        
    end

    function update_lambda(docs)

    end

    function approx_bound(docs, gamma)

    end
end
