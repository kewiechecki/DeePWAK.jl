@doc raw"""
DEWAKache(D,K,L)

Cached distance matrices and neighbors for a `DEWAKSS` model.

See also: `DEWAKSS`.
"""
struct DEWAKache
    D :: AbstractArray
    K :: AbstractArray
    L :: AbstractArray
end

@doc raw"""
`DEWAKSS`



See also: `DeePWAK`, `Autoencoders.AbstractDDAE`
"""
struct DEWAKSS <: Autoencoders.AbstractDDAE
    autoencoder::Autoencoders.AbstractEncoder
    metric
    pca
    d_max::Integer
    k_max::Integer
    d::Integer
    k::Integer
    dat
    cache::DEWAKache
end
@functor autoencoder, metric

function DEWAKSS(X::AbstractMatrix)
    pca = fit(PCA,X)
    E = predict(pca,X)

    d_max,k_max = size(E)
    d = d_max ÷ 2

    D = mapreduce(d->pcadist(E,d),zcat,1:d_max)
    K = mapreduce(d->perm(d,k_max),zcat,D)
    
    encoder = Autoencoder(identity,identity)
    cache = DEWAKache(D,K)
    DEWAKSS(encoder,inveucl,pca,d_max,k_max,d,d)
end

function dist(M::DEWAKSS,E::AbstractMatrix)
    M.metric(predict(M.pca,E))
end

function dist(M::DEWAKSS,d::Union{Integer,UnitRange})
    M.cache.D[:,:,d]
end

function knn(M::DEWAKSS,D::AbstractMatrix{<:AbstractFloat})
    K = perm(D,M.k)
    knn(K,M.k)
end

function knn(M::DEWAKSS,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    knn(M.cache.K[:,:,d],k)
end

function kern(M::DEWAKSS)
    kern(M,M.d,M.k)
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

function (M::DEWAKSS)(X)
    decode(M,diffuse(M,X))
end

function (M::DEWAKSS)()
    decode(M,diffuse(M))
end
