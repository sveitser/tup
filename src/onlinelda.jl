require("Distributions", "Winston")
using(Distributions)

global const meanchangethresh = 0.001

function dirichlet_expecation(alpha)
    psigamma(alpha) - psigamma(sum(alpha))
end

type OnlineLDA
    OnlineLDA(vocab, K, D, alpha, eta, tau0, kappa)

    lambda = rand(Gamma(100, 1 / 100), K, W)
    Elogbeta = dirichlet_expecation(lambda)
    expElogbeta = exp(Elogbeta)

    function parse_doc_list(docs)
        D = size(docs, 1)
        wordids = Array(Array{Int64}, 0)
        wordcts = Array(Array{Int64}, 0)
        for d in 1:D
            docs[d] = docs[d].lowercase()
            docs[d] = replace(docs[d],r"-"," ")
            docs[d] = replace(docs[d],r"[^a-z ]","")
            docs[d] = replace(docs[d],r" +"," ")
            words = split(docs[d])
            ddict = Dict{Int64, Int64}()
            for word in words
                if has(vocab, word)
                    wordtoken = vocab[word]
                    if !has(ddict, wordtoken)
                        ddict[wordtoken] = 0
                    end
                    ddict[wordtoken] += 1
                end
            end
            push(wordids, keys(ddict))
            push(wordcts, values(ddict))
        end
        return wordids, wordcts
    end

    function do_e_step(docs::String)
        do_e_step([docs])
    end

    function do_e_step(docs)
        wordids, wordcts = parse_doc_list(docs, vocab)
        batchD = size(docs, 1)
      
        gamma  = rand(Gamma(100, 1 / 100, batchD, K))
        Elogtheta = dirichlet_expecation(gamma)
        expElogtheta = exp(Elogbeta)

        sstats = zeros(size(lambda))
        meanchange = 0
        for d in 1:batchD
            phinorm = expElogtheta[d, :] * expElogbeta[d, :]' + 1e-100
            for it in 1:100
                lastgamma = gamma[d, :]
                gamma[d, :] = alpha + expElogbeta[d, :] * wordcts[d] / phinorm  *
                    expElogbeta[d, :]
                Elogtheta[d, :] = dirichlet_expecation(gamma[d, :])
                phinorm = expElogtheta[d, :] * expElogbeta[d, :]' + 1e-100
                meanchage = mean(abs(gamma[d, :] - lastgamma))
                if meanchange < meanchangethresh
                    break
                end
            end
            sstats[:, wordids[d]] = expElogtheta[:, d]' * wordcts[d] / phinorm
        end
        sstats *= expElogbeta
        return gamma, sstats
    end

    function update_lambda(docs)
        rhot = (tau0 + updatect)^( - kappa)
        gamma, sstats = do_e_step(docs)
        bound = approx_bound(docs, gamma)
        lambda = lambda * (1 - rhot) 
            + rhot * (eta + D * sstats / size(docs, 1))
        Elogbeta = dirichlet_expecation(lambda)
        expElogbeta = exp(Elogbeta)
        updatect += 1
        return gamma, bound
    end

    function approx_bound(docs::String, gamma)
        approx_bound([docs], gamma)
    end

    function approx_bound(docs, gamma)
        wordids, wordcts = parse_doc_list(docs, gamma)
        batchD = size(docs, 1)
        
        score = 0
        Elogtheta = dirichlet_expecation(gamma)
        expElogtheta = exp(Elogtheta)
        
        for d in 1:batchD
            ids = wordids[d]
            phinorm = zeros(length(ids))
            for i in 1:length(length(ids))
                temp = Elogtheta[d, :] + Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = log(sum(exp(temp - tmax))) + tmax
            end
            score += sum(cts * phinorm)
        end
        score += sum((alpha - gamma) * Elogbeta)
        score += sum(lgamma(gamma) - lgamma(alpha))
        score += sum(lgamma(alpha * K) - lgamma(sum(gamma, 1)))

        score *= D / size(docs, 1)

        score += sum((eta - lambda) * Elogbeta)
        score += sum(lgamma(lambda) - lgamma(eta))
        score += sum(lgamma(eta * W) - lgamma(sum(lambda, 1)))

        return score
    end
end
