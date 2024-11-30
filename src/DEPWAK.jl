@doc raw"""
`AbstractDeePWAK <: AbstractDEWAK`

Should have field `dewak`.

See also: `DEPWAK`, `DDAEWAK`, `AbstractDEWAK`.
"""
abstract type AbstractDeePWAK <: AbstractDEWAK
end

function dist(M::AbstractDeePWAK,E)
    dist(M.dewak,E)
end

function dist(M::AbstractDeePWAK)
    dist(M.dewak)
end

function knn(M::AbstractDeePWAK,D)
    knn(M.dewak,D)
end

function knn(M::AbstractDeePWAK)
    knn(M.dewak)
end

function losslog(M::AbstractDeePWAK)
    losslog(M.dewak)
end

@doc raw"""
`DEPWAK(autoencoder::DEWAK,γ::AbstractFloat)`

Model for Denoising with deep learning of a Partitioned Weighted Affinity Kernel. 

See also: `DEWAK`, `Autoencoders.AbstractPartitioned`, `Leiden.leiden`
"""
struct DEPWAK <: AbstractDeePWAK
    dewak :: AbstractDEWAK
    γ::AbstractFloat
    clusts
    partition
    cache
end
@functor DEPWAK (dewak,)

function dist(M::DEPWAK,E)
    M.dewak.metric(predict(M.dewak.pca,E))
end

function knn(M::DEPWAK,D)
    K = perm(D,M.dewak.k)
    knn(K,M.dewak.k)
end

function cluster(M::DEPWAK,G::AbstractMatrix)
    leiden(G,resolution_parameter=M.γ,n_iterations=-1)
end

function (M::DEPWAK)(X)
    decode(M,diffuse(M,X))
end

