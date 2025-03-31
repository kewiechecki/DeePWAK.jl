function rep(expr,n)
    return map(_->expr,1:n)
end

function perm(D::AbstractMatrix;dims=1,rev=true)
    n = size(D,1)
    K = sortperm(D,dims=dims,rev=rev) .% n
    K[K .== 0] .= n
    return K
end

#sortperm(d) is slower than partialsortperm(d,1:size(d,1)) and I don't know why
function perm(D::AbstractMatrix,k_max::Integer;dims=1,rev=true)
    mapslices(d->partialsortperm(d,1:k_max,rev=rev),
              D,dims=dims)
end

function knn(N::AbstractMatrix,k::Integer)
    _,n = size(N)
    N = N[1:k,:]
    J = reshape(N,:)
    I = mapreduce(i->rep(i,k),vcat,1:n)
    return sparse(I,J,1,n,n)
end
                  
function clusts(N,k,γ)
    K = knn(N,k)
    C = Leiden.leiden(K,"mod++",γ=γ)
    return C
end

function partitionmat_slow(C)
    (sum ∘ map)(unique(C)) do c
        x = C .== c
        return x * x'
    end
end

function partitionmat(C)
    v = onehotbatch(C,unique(C))
    v' * v
end

function loss_clust(X,F,D,N,k,γ,s,M_sparse,M_dense)
    C = clusts(N,k,γ)
    P = partitionmat(C) |> gpu
    G = wak(D .* P)
    F̂ = ((G^s) * F')'
    Ê = M_sparse.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end

function zcat(args...)
    cat(args...;dims=3)
end
        
# modularity score
# supports probabilistic cluster assignment
# accepts potentially weighted adjacency matrix G,
# ptentially weighted partition matrix P, resolution γ
function modularity(G::AbstractMatrix, P::AbstractMatrix, γ::AbstractFloat)
    Ĝ = P .* G
    Ĝ_inv = (1 .- P) .* G
    m = sum(G)
    k_v = sum(Ĝ; dims = 2)
    k_e = sum(Ĝ; dims = 1)
    e = sum(k_e; dims = 2)
    μ = mean(e)
    K = sum(k_v; dims = 2)
    H = 1 / (2 * μ) * sum(e .- γ .* K .^ 2 ./ (2 * μ))
    return H
end

function modularity(G::AbstractMatrix, P::AbstractMatrix, γ::AbstractFloat)
    k_v = sum(G; dims = 1)
    m = sum(k_v)
    K = γ .* k_v * k_v'
    B = G .- (K ./ (2 * m))
    tr(P' * B * P)
end

function silhouette(D::AbstractMatrix, P::AbstractMatrix;
                    inverse = true)
    if inverse
        D = 1 ./ D
    end
    G = D .* P
    G_inv = D .* (1 .- P)
    d_inter = maximum(G; dims = 2)
    d_intra = minimum(G_inv; dims = 2)
    sil = (d_inter .- d_intra) ./ maximum(hcat(d_inter, d_intra); dims = 2)
    mean(sil)
end

