function update!(M::DEWAK, loss, window_d, window_k)
    #d_max = length(cache(M)[:d])
    #k_max = size(cache(M)[:k][1],1)
    d_max,k_max = size(M.pcs)

    G_0 = kern(M)
    v_d = maximum([1, (M.d - window_d)]):minimum([M.d + window_d, d_max])
    v_k = maximum([1, (M.k - window_k)]):minimum([M.k + window_k, k_max])

    kern_d = d->begin
        D = dist(M.cache, d)
        push!(M.cache.dict[:dist], D)
        kern(M.cache, D, d, M.k)
    end

    kern_k = k->begin
        G = knn(M.cache, M.d, k)
        push!(M.cache.dict[:graph], G)
        kern(M.dist, G)
    end
    
    #params_d,L_d = updateparam!(M,:d,loss,
    #                            d->kern(M,d,M.k),v_d)
    params_d, L_d = updateparam!(M, :d, loss, kern_d, v_d)
    M.dist = M.cache.dict[:dist][argmin(L_d[:, 1])]
    
    #params_k,L_k = updateparam!(M,:k,loss,
    #                   k->kern(M,M.d,k),v_k)
    params_k,L_k = updateparam!(M,:k, loss, kern_k, v_k)
    M.graph = M.cache.dict[:graph][argmin(L_k[:, 1])]

    M.cache.dict[:dist] = []
    M.cache.dict[:graph] = []

    return vcat(params_d,params_k), vcat(L_d, L_k)
end

function updateparam!(M::AbstractDEWAK, id_param, f_loss, f_kern, v_param)
    n_v = length(v_param)
    v_G = map(f_kern,v_param)
    L = mapreduce(G->hcat(f_loss(G)...), vcat, v_G)

    tab = vcat(rep(params(M), n_v)...)
    tab[:, id_param] = v_param

    losslabs = losslabels(M.cache)
    losstab = DataFrame(L, losslabs)

    updateloss!(M, tab, losstab)
    set!(M, id_param, v_param[argmin(L[:, 1])])
    return tab, losstab
end

function update_d!(M::DEWAK, loss, d)
    n_d = length(d)

    G = map(d_i->kern(M, d_i, M.k), d)
    
    L_d = map(loss, G)
    L_d = DataFrame(d = d, k = M.k, loss = L_d)
    L_d = hcat(d, rep(M.k, n_d), L_d)

    M.d = d[argmin(L_d[:, 3])]
    M.cache.loss = vcat(M.cache.loss, L_d)
    return L_d
end

function update_k!(M::DEWAK, loss, k)
    n_k = length(k)

    G = map(k_i->kern(M, M.d, k_i), k)
    
    L_k = map(loss, G)
    L_k = hcat(rep(M.d, n_k), k, L_k)
    
    M.k = k[argmin(L_k[:, 3])]
    M.cache.loss = vcat(M.cache.loss, L_k)
    return L_k
end

function update!(M::DEPWAK, loss, window_d, window_k, window_γ, n_γ)
    v_γ = vcat(M.γ, rand(Uniform(relu(M.γ - window_γ),
                                 M.γ + window_γ), n_γ))

    P_0 = partition(M)
    #loss_dk = G->(loss ∘ wak)(G .* P_0)

    params_dk,L_dk = update!(M.dewak, loss, window_d, window_k)
    params_dk[:, :γ] .= M.γ
    params_dk[:, :n_clusts] .= maximum(M.clusts)
    updateloss!(M, params_dk, L_dk)

    M.cache.dict[:clusts] = []
    M.cache.dict[:P] = []

    #n_dk, n_col = size(L_dk)
    #L_dk = hcat(L_dk[:, 1:(n_col - 1)],
    #            rep(M.γ, n_dk),
    #            L_dk[:, n_col])

    G = dist(M) .* knn(M)
    #loss_P = P->(loss ∘ wak)(G .* P)

    #C = map(γ_i->cluster(M, G, γ_i), γ)
    #P = map(partitionmat, C) 

    #graph = M.graphfn(dist(M), knn(M))
    g = graph(M)

    kern_γ = γ->begin
        C = cluster(M, g, γ)
        push!(M.cache.dict[:clusts], C)
        P = partitionmat(C)
        push!(M.cache.dict[:P], P)
        wak(G .* P)
    end
    
    #L_γ = map(loss_γ,P)
    params_γ, L_γ = updateparam!(M, :γ, loss, kern_γ, v_γ)

    n_clusts = maximum.(M.cache.dict[:clusts])
    params_γ.n_clusts .= n_clusts
    updateloss!(M, params_γ, L_γ)
    i = argmin(L_γ[:, 1])

    #M.γ = v_γ[i]
    #M.n_clusts = n_clusts[i]
    M.clusts = M.cache.dict[:clusts][i]
    M.partition = M.cache.dict[:P][i]
    M.cache.dict[:clusts] = []
    M.cache.dict[:P] = []
    #L_γ = hcat(rep(M.dewak.d, n_γ),
    #           rep(M.dewak.k, n_γ),
    #           γ, L_γ)
    L = vcat(L_dk, L_γ)
    params = vcat(params_dk, params_γ)
    #M.loss = vcat(M.loss, L)
    return params, L
end 

function update!(M::DDAEWAK, loss, opt, args...)
    G_0 = kern(M) |> gpu
    X = data(M) |> gpu
    #loss_dec = m->Flux.mse(decode(m, (G * encode(m, X)')'),X)
    loss_dec = m->loss(m, decode(m, (G * encode(m, X)')'), X)
    
    L = update!(M.autoencoder, loss, opt)

    M.epoch = M.epoch + 1
    tab = params(M)
    losstab = DataFrame(hcat(L...), losslabels(M.cache))
    updateloss!(M, tab, losstab)

    E = encode(M, X) |> cpu
    
    loss_wak = (G)->loss(decode(M, wak((G * E')')), X)
    M.dewak.dat = encode(M,data(M))
    L_dk = update!(M.dewak, G_i->loss(M, G_i), window_d, window_k)
    L = vcat(L, L_dk)
    return L
end

    
