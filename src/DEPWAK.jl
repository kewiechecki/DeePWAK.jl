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

@doc raw"""
`DEPWAK(autoencoder::DEWAK,γ::AbstractFloat)`

Model for Denoising with deep learning of a Partitioned Weighted Affinity Kernel. 

See also: `DEWAK`, `Autoencoders.AbstractPartitioned`, `Leiden.leiden`
"""
mutable struct DEPWAK <: AbstractDeePWAK
    dewak :: AbstractDEWAK
    graphfn
    clustfn
    γ :: AbstractFloat
    clusts :: AbstractArray
    partition :: AbstractMatrix
    cache :: DEWAKache
end
@functor DEPWAK (dewak,)

function DEPWAK(dewak::DEWAK, clustfn;
                graphfn = (D,G)->(D .* G),
                γ_0 = eps(), losslabs=[:mse])
    clusts = clustfn(graphfn(knn(dewak), dist(dewak)),γ_0)
    partition = partitionmat(clusts)

    id_params = vcat((names ∘ params)(dewak),"γ","n_clusts")
    n_params = length(id_params)
    params_0 = DataFrame(Array{Float64}(undef,0,n_params),id_params)

    n_loss = length(losslabs)
    L_0 = DataFrame(Array{Float64}(undef,0,n_loss),
                    losslabs)

    cache = DEWAKache(Dict([(:clusts,[]),
                            (:P,[])]),
                      params_0,L_0)

    DEPWAK(dewak,graphfn,clustfn,γ_0,clusts,partition,cache)
end

function DEPWAK(X::AbstractArray, clustfn;
                graphfn = (D,G)->(D .* G),
                γ_0 = eps(),
                metric=inveucl,
                losslabs=[:mse], kwargs...)
    dewak = DEWAK(X; metric=metric,losslabs=losslabs,kwargs...)

    DEPWAK(dewak, clustfn;
           graphfn=graphfn, γ_0=γ_0,
           losslabs=losslabs)
end

function params(M::DEPWAK)
    hcat(params(M.dewak),DataFrame(γ=M.γ,n_clusts=maximum(M.clusts)))
end

function set_γ!(M::DEPWAK,γ)
    M.γ = γ
end

function set!(M::DEPWAK,param,val)
    @match param begin
        :γ => set_γ!(M,val)
        :gamma => set_γ!(M,val)
        :resolution => set_γ!(M,val)
        :loss => updateloss!(M,val)
        _ => set!(M.dewak,param,val)
    end
end

function data(M::DEPWAK)
    data(M.dewak)
end

function encode(M::DEPWAK,X)
    X
end

function decode(M::DEPWAK,X)
    X
end
function dist(M::DEPWAK,E)
    M.dewak.metric(predict(M.dewak.pca,E))
end

function dist(M::DEPWAK,d::Integer)
    dist(M.dewak,d)
end

function dist(M::DEPWAK)
    dist(M.dewak)
end

function knn(M::DEPWAK,D)
    K = perm(D,M.dewak.k)
    knn(K,M.dewak.k)
end

function knn(M::DEPWAK,d::Integer,k::Integer)
    knn(M.dewak,d,k)
end

function cluster(M::DEPWAK,G,#::AbstractMatrix,
                 γ::AbstractFloat)
    M.clustfn(G,γ)
end

function cluster(M::DEPWAK,d::Integer,k::Integer,γ::AbstractFloat)
    G = knn(M,d,k) .* dist(M,d)
    cluster(M, G, γ)
end

function cluster(M::DEPWAK)
    M.clusts
end

function partition(M::DEPWAK, clusts)
    partitionmat(clusts)
end

function partition(M::DEPWAK)
    M.partition
end

function kern(M::DEPWAK, d::Integer, k::Integer, γ::AbstractFloat)
    G = dist(M,d) .* knn(M,d,k)
    P = partition(M, cluster(M,G,γ))
    wak(G .* P)
end

function kern(M::DEPWAK)
    wak(dist(M) .* knn(M) .* partition(M))
end

function (M::DEPWAK)(X)
    decode(M,diffuse(M,X))
end

