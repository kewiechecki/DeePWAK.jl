@doc raw"""
`DEWAK`

Denoising with a Weighted Affinity Kernel [originally for Single cell Sequencing].

See also: `DeePWAK`, `Autoencoders.AbstractDDAE`
"""
struct DEWAK <: Autoencoders.AbstractDDAE
    autoencoder::Autoencoders.AbstractEncoder
    metric
    pca
    d::Integer
    k::Integer
    γ::AbstractFloat
    dist::AbstractMatrix
    graph::AbstractMatrix
    partition::AbstractMatrix
    kern::AbstractMatrix
    dat
    cache::DEWAKache
end
@functor DEWAK (autoencoder, metric)

function DEWAK(X::AbstractMatrix)
    pca = fit(PCA,X)
    E = predict(pca,X)

    d_max,k_max = size(E)
    d = d_max ÷ 2

    D = mapreduce(d->pcadist(E,d),zcat,1:d_max)
    K = mapreduce(d->perm(d,k_max),zcat,D)
    
    encoder = Autoencoder(identity,identity)
    cache = DEWAKache(D,K)
    DEWAK(encoder,inveucl,
          d,d,1.0,
          D_0,G_0,P_0,
          dat,pca,cache)
end

function dist(M::DEWAK,E::AbstractMatrix)
    M.metric(predict(M.pca,E))
end

function dist(M::DEWAK,d::Union{Integer,UnitRange})
    M.cache.D[:,:,d]
end

function knn(M::DEWAK,D::AbstractMatrix{<:AbstractFloat})
    K = perm(D,M.k)
    knn(K,M.k)
end

function knn(M::DEWAK,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    knn(M.cache.K[:,:,d],k)
end

function kern(M::DEWAK)
    kern(M,M.d,M.k)
end

function kern(M::DEWAK,E::AbstractMatrix)
    D = dist(M,E)
    G = knn(M,D)
    wak(G .* D)
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

function diffuse(M::DEWAK)
    (kern(M) * encode(M)')'
end

function (M::DEWAK)(X)
    decode(M,diffuse(M,X))
end

function (M::DEWAK)()
    decode(M,diffuse(M))
end
