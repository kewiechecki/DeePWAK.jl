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



See also: `DEWAKSS`, `Autoencoders.AbstractPartitioned`, `Leiden.leiden`
"""
struct DeePWAK <: AbstractPartitioned
    dewak::DEWAKSS
    γ::AbstractFloat
end
@functor dewak

function dist(M::DeePWAK,E)
    M.dewak.metric(predict(M.dewak.pca,E))
end

function knn(M::DeePWAK,D)
    K = perm(D,M.dewak.k)
    knn(K,M.dewak.k)
end

function kern(M::DeePWAK,E)
    D = dist(M,E)
    G = knn(M,D)
    G = G .* D
    wak(G)
end

function (M::DeePWAK)(X)
    decode(M,diffuse(M,X))
end

end # module DeePWAK
