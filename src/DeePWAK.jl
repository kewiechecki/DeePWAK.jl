module DeePWAK

using Reexport
using Flux, Functors
@reexport using Autoencoders
using Distributions, MultivariateStats, SparseArrays
@reexport using Leiden

export DEWAKSS, DeePWAK
export dist, knn, kern
export perm, clusts, partitionmat
export update!

include("clustering.jl")

include("DEWAKSS.jl")

@doc raw"""
`DeePWAK(autoencoder::DEWAKSS,γ::AbstractFloat)`

Model for Denoising with deep learning of a Partitioned Weighted Affinity Kernel. 

See also: `DEWAKSS`, `Autoencoders.AbstractPartitioned`, `Leiden.leiden`
"""
struct DEPWAK <: AbstractPartitioned
    dewak :: DEWAKSS
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

function kern(M::DEPWAK,E)
    D = dist(M,E)
    G = knn(M,D)
    G = G .* D
    wak(G)
end

function (M::DEPWAK)(X)
    decode(M,diffuse(M,X))
end

end # module DeePWAK
