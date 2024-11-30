@doc raw"""
DEWAKache(D,K,L)

Cached distance matrices and neighbors for a `DEWAKSS` model.

See also: `DEWAKSS`.
"""
struct DEWAKache
    D :: AbstractVector
    K :: AbstractVector
    L :: AbstractArray
end

@doc raw"""
`DEWAKSS`

Denoising with a Weighted Affinity Kernel [originally for] Single cell Sequencing.

See also: `DeePWAK`, `Autoencoders.AbstractDDAE`
"""
struct DEWAKSS <: Autoencoders.AbstractDDAE
    autoencoder::Autoencoders.AbstractEncoder
    metric
    d::Integer
    k::Integer
    graph::AbstractMatrix
    kern::AbstractMatrix
    dat
    pca
    cache::DEWAKache
end
@functor DEWAKSS (autoencoder, metric)

function DEWAKSS(X::AbstractMatrix; metric=inveucl)
    pca = fit(PCA,X)
    E = predict(pca,X)

    d_max,k_max = size(E)
    d = d_max ÷ 2
    k = d

    D = map(d->metric(E[1:d,:]),1:d_max)
    K = map(d->perm(d,k_max),D)
    G = knn(K[d],k)
    
    L_0 = Array{Float64}(undef,0,4)
    
    encoder = Autoencoder(identity,identity)
    cache = DEWAKache(D,K,L_0)
    
    DEWAKSS(encoder, metric,
            d, k,
            G, wak(G .* D[d]), X,
            pca,cache)
end

function dist(M::DEWAKSS,E::AbstractMatrix)
    M.metric(predict(M.pca,E))
end

function dist(M::DEWAKSS,d::Union{Integer,UnitRange})
    M.cache.D[d]
end

function knn(M::DEWAKSS)
    M.graph
end

function knn(M::DEWAKSS,D::AbstractMatrix{<:AbstractFloat})
    K = perm(D,M.k)
    knn(K,M.k)
end

function knn(M::DEWAKSS,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    knn(M.cache.K[d],k)
end

function kern(M::DEWAKSS)
    M.kern
end

function kern(M::DEWAKSS,E::AbstractMatrix)
    D = dist(M,E)
    G = knn(M,D)
    wak(G .* D)
end

function kern(M::DEWAKSS,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    D = dist(M,d)
    G = knn(M,k)
    wak(G .* D)
end

function encode(M::DEWAKSS)
    encode(M,M.dat)
end

function diffuse(M::DEWAKSS)
    (kern(M) * encode(M)')'
end

function loss(M::DEWAKSS)
    M.cache.loss
end

function (M::DEWAKSS)(X)
    decode(M,diffuse(M,X))
end

function (M::DEWAKSS)()
    decode(M,diffuse(M))
end
