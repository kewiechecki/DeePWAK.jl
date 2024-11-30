@doc raw"""
`AbstractDEWAK <: AbstractDDAE`

Should be callable and implement `data`, `encode`, `knn`, `dist`, `pca`.

See also: `DEWAK`, `DDAEWAK`, `AbstractDDAE`, `AbstractEncoder`
"""
abstract type AbstractDEWAK <: AbstractDDAE
end

function loss(M::AbstractDEWAK)
    Flux.mse(M(), data(M))
end

function kern(M::AbstractDEWAK,E::AbstractMatrix)
    D = dist(M,E)
    G = knn(M,D)
    wak(G .* D)
end

function diffuse(M::AbstractDEWAK)
    E = encode(M)
    (kern(M) * E')'
end

@doc raw"""
DEWAKache(D,K,L)

Cached distance matrices and neighbors for a `DEWAK` model.

See also: `DEWAK`.
"""
struct DEWAKache
    D :: AbstractVector
    K :: AbstractVector
    L :: AbstractArray
end

@doc raw"""
`DEWAK`

Denoising with a Weighted Affinity Kernel [originally for] Single cell Sequencing.

See also: `DeePWAK`, `Autoencoders.AbstractDDAE`
"""
struct DEWAK <: AbstractDEWAK
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
@functor DEWAK (autoencoder, metric)

function DEWAK(X::AbstractMatrix; metric=inveucl)
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
    
    DEWAK(encoder, metric,
            d, k,
            G, wak(G .* D[d]), X,
            pca,cache)
end

@doc raw"""
`data(M::DEWAK) -> typeof(M.dat)`

Accessor for `M.dat`.

See also : `DEWAK`.
"""
function data(M::DEWAK)
    M.dat
end

function encode(M::DEWAK,X)
    X
end

function decode(M::DEWAK,X)
    X
end

function pca(M::DEWAK, X)
    predict(M.pca,E)
end
    
function dist(M::DEWAK,X::AbstractMatrix)
    M.metric(pca(M,X))
end

function dist(M::DEWAK,d::Union{Integer,UnitRange})
    M.cache.D[d]
end

function knn(M::DEWAK)
    M.graph
end

function knn(M::DEWAK,D::AbstractMatrix{<:AbstractFloat})
    K = perm(D,M.k)
    knn(K,M.k)
end

function knn(M::DEWAK,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    knn(M.cache.K[d],k)
end

function kern(M::DEWAK)
    M.kern
end

function kern(M::DEWAK,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    D = dist(M,d)
    G = knn(M,k)
    wak(G .* D)
end

function encode(M::DEWAK)
    encode(M,M.dat)
end

function losslog(M::DEWAK)
    M.cache.loss
end

function (M::DEWAK)(X)
    decode(M,diffuse(M,X))
end

function (M::DEWAK)()
    decode(M,diffuse(M))
end
