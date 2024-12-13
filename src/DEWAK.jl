@doc raw"""
`AbstractDEWAK <: AbstractDDAE`

Should be callable and implement `data`, `encode`, `knn`, `dist`, `pca`.
Should have field `cache::DEWAKache`.

See also: `DEWAK`, `DDAEWAK`, `AbstractDDAE`, `AbstractEncoder`
"""
abstract type AbstractDEWAK <: AbstractDDAE
end

function loss(M::AbstractDEWAK)
    Flux.mse(M(), data(M))
end

#function loss(M::AbstractDEWAK,G::AbstractMatrix)
#    E = encode(M,data(M))
#    Ê = (G * E')'
#    Flux.mse(decode(M,Ê),data(M))
#end

function loss(M::AbstractDEWAK,args...)
    G = foldl(.*,[args...])
    E = encode(M,data(M))
    Ê = (wak(G) * E')'
    Flux.mse(decode(M,Ê),data(M))
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

function cache(M::AbstractDEWAK)
    M.cache.dict
end

function updateloss!(M::AbstractDEWAK,params::DataFrame,loss::DataFrame)
    updateloss!(M.cache,params,loss)
end

function losslog(M::AbstractDEWAK)
    hcat(M.cache.params,M.cache.loss)
end

@doc raw"""
DEWAKache(D,K,L)

Cached distance matrices and neighbors for a `DEWAK` model.

See also: `DEWAK`.
"""
mutable struct DEWAKache
    dict :: Dict
    params :: DataFrame
    loss :: DataFrame
end

function losslabels(cache::DEWAKache)
    names(cache.loss)
end

function updateloss!(cache::DEWAKache,params::DataFrame,L::DataFrame)
    cache.params = vcat(cache.params,params)
    cache.loss = vcat(cache.loss,L)
end

@doc raw"""
`DEWAK`

Denoising with a Weighted Affinity Kernel [originally for] Single cell Sequencing.

See also: `DeePWAK`, `Autoencoders.AbstractDDAE`
"""
mutable struct DEWAK <: AbstractDEWAK
    metric
    d::Integer
    k::Integer
    pcs::AbstractMatrix
    dist::AbstractMatrix
    graph::AbstractMatrix
    kern::AbstractMatrix
    dat
    pca
    cache::DEWAKache
end
@functor DEWAK (autoencoder, metric)

function updatecache!(M::DEWAK, X::AbstractMatrix)
    M.dat = X
    M.pca = fit(PCA,X)
    M.pcs = predict(M.pca,X)

    d_max,k_max = size(E)
    M.d = minimum(M.d,d_max)
    M.k = minimum(M.k,k_max)
    
    M.dist = metric(M.pcs[1:M.d,:])

    K = perm(M.dist,M.k)

    M.graph = knn(K,M.k)
    M.kern = kern(M,M.graph) 
end

function DEWAK(X::AbstractMatrix;
               metric=inveucl, losslabs=[:mse],
               d_0 = 1, k_0 = 1)
    pca = fit(PCA,X)
    E = predict(pca,X)

    d_max,k_max = size(E)
    #d = d_max ÷ 2
    #k = d

    #D = metric(E[1:d_0,:])
    #K = perm(D,k_0)
    #G = knn(K,k_0)
    v_D = map(d->metric(E[1:d,:]),1:d_max)
    v_K = map(D->perm(D,k_max),v_D)

    D = v_D[d_0]
    G = knn(v_K[d_0],k_0)
    
    params_0 = DataFrame(Array{Integer}(undef,0,2),[:d,:k])
    n_loss = length(losslabs)
    L_0 = DataFrame(Array{Float64}(undef,0,n_loss),
                    losslabs)

    dict = Dict([(:d,v_D),(:k,v_K),
                 (:dist,[]),(:graph,[])])
    cache = DEWAKache(dict,params_0,L_0)
    
    DEWAK(metric, d_0, k_0, E, D, G,
          wak(G .* D), X,
          pca,cache)
end

@doc raw"""
`params(M::DEWAK) -> DataFrame`
    """
function params(M::DEWAK)
    DataFrame(d=M.d,k=M.k)
end

function set_d!(M::DEWAK,d::Integer)
    M.d = d
end

function set_k!(M::DEWAK,k::Integer)
    M.k = k
end

@doc raw"""
`set!(M::AbstractDEWAK, param::Symbol,
      value::typeof(params(M)[1,param]) -> Nothing
"""
function set!(M::DEWAK, param::Symbol, val)
    @match param begin
        :d => set_d!(M,val)
        :k => set_k!(M,val)
        :loss => updateloss!(M,val...)
        _ => printf(string(typeof(M))*" lacks parameter "*string(param))
    end
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
    predict(M.pca,X)
end
    
function dist(M::DEWAK,X::AbstractMatrix)
    M.metric(pca(M,X))
end

function dist(M::DEWAK,d::Integer)#d::Union{Integer,UnitRange})
    #cache(M)[:d][d]
    M.metric(M.pcs[1:d,:])
end

function dist(M::DEWAK)
    #cache(M)[:d][M.d]
    M.dist
end

function dist(cache::DEWAKache,d::Union{Integer,UnitRange})
    cache.dict[:d][d]
end

function knn(M::DEWAK)
    M.graph
end

function knn(M::DEWAK,D::AbstractMatrix{<:AbstractFloat})
    K = perm(D,M.k)
    knn(K,M.k)
end

function knn(M::DEWAK,D::AbstractMatrix{<:AbstractFloat},k::Integer)
    K = perm(D,k)
    knn(K,k)
end

function knn(M::DEWAK,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    #knn(cache(M)[:k][d],k)
    D = dist(M,d)
    knn(M,D,k)
end

function knn(M::DEWAK,k::Integer)
    #knn(cache(M)[:k][M.d],k)
    knn(M,M.d,k)
end

function knn(cache::DEWAKache,
             d::Union{Integer,UnitRange},
             k::Union{Integer,UnitRange})
    knn(cache.dict[:k][d],k)
end

function knn(M::DEWAK,cache::DEWAKache,k::Integer)
    knn(cache.dict[:k][M.d],k)
end

function kern(D::AbstractMatrix,G::AbstractMatrix)
    wak(D .* G)
end

function kern(M::DEWAK,D::AbstractMatrix,k::Integer)
    wak(D .* knn(M,D,k))
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

function kern(cache::DEWAKache,
              D::AbstractMatrix,
              d::Union{Integer,UnitRange},
              k::Union{Integer,UnitRange})
    G = knn(cache,d,k)
    wak(G .* D)
end

function kern(cache::DEWAKache,
              d::Union{Integer,UnitRange},
              k::Union{Integer,UnitRange})
    D = dist(cache,d)
    kern(cache,D,d,k)
end

function encode(M::DEWAK)
    encode(M,M.dat)
end

function (M::DEWAK)(X)
    decode(M,diffuse(M,X))
end

function (M::DEWAK)()
    decode(M,diffuse(M))
end
