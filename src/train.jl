function update!(M::DEWAKSS, loss, window_d, window_k)
    d_0 = M.d
    k_0 = M.k

    G_0 = kern(M)
    loss_enc = M->begin
        E = encode(
    update!(M.autoencoder,loss)
    d = maximum([1,(d_0 - window_d)]):minimum([d_0 + window_d,M.d_max])
    k = maximum([1,(k_0 - window_k)]):minimum([k_0 + window_k,M.k_max])

    update_d!(M,loss,d)
    update_k!(M,loss,k)
end

function update_d!(M::DEWAKSS,loss,d)
    n_d = length(d)

    G = map(d_i->kern(M,d_i,M.k),d)
    
    L_d = map(loss,G)
    L_d = hcat(d,rep(k_0,n_d),L_d)

    M.d = d[argmin(L_d[:,3])]
    push!(M.cache.loss,L_d)
end

function update_k!(M::DEWAKSS,loss,k)
    n_k = length(k)

    G = map(k_i->kern(M,M.d,k_i),k)
    
    L_k = map(loss,G)
    L_k = hcat(rep(d,n_k),k,L_k)
    
    M.k = k[argmin(L_k[:,3])]
    push!(M.cache.loss,L_k)
end

    
