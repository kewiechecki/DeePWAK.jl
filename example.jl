using DeePWAK

using PyCall
@pyimport leidenalg
@pyimport igraph

using RCall
@rimport fgsea

function pygraph(D,G)
    igraph.Graph[:Weighted_Adjacency](D .* G)
end

function pyleiden(graph, γ, args...; kwargs...)
    clusts =leidenalg.find_partition(graph,
                             leidenalg.RBConfigurationVertexPartition,
                             args...;
                             n_iterations = -1,
                             resolution_parameter=γ, kwargs...)
    clusts.membership .+ 1
end

function jleiden(G,γ)
    leiden(G,"cpm";γ=γ)
end

dat = (DataFrame ∘ CSV.File)("exampledat/z_dat.csv",normalizenames=true);

# Preprocessing

# remove row names
dat = Matrix(dat[:,2:end]);

# scale data such that all values are between -1 and 1 
X = scaledat(dat')
m,d = size(X)

# initialize model
dewak = DEWAK(X; d_0 = 6, k_0 = 6)

window_d = 5
window_k = 5
window_γ = 1.0
n_γ = 5

steps = 100

L_dewak = @showprogress mapreduce(vcat,1:steps) do _
    update!(dewak,G_i->loss(dewak,G_i),
            window_d,window_k)
end

depwak = DEPWAK(dewak, pyleiden; graphfn=pygraph)

L_depwak = @showprogress mapreduce(vcat,1:steps) do _
    update!(depwak,G->loss(depwak,G),
            window_d,window_k,window_γ,n_γ)
end

sae = SAE(m,4*m,relu) |> gpu
encoder = Chain(Dense(m => m, tanh), X->encode(sae,X)) |> gpu
decoder = Chain(X->decode(sae,X),Dense(m => m, tanh)) |> gpu
autoenc = Autoencoder(encoder, decoder)

path = "tmp/" #where to save models
epochs = 1000
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

# sparsity coefficient
α = 0.001

loader = Flux.DataLoader((X,X),batchsize=batchsize,shuffle=true)

L_dec = train!(autoenc,path,loss(Flux.mse),loader,opt,epochs;
               savecheckpts=false)

E = encode(autoenc,X |> gpu)

dewak = DEWAK(E |> cpu)
L_dewak = @showprogress mapreduce(vcat,1:steps) do _
    update!(dewak,G_i->loss(dewak,G_i),
            window_d,window_k)
end


ddaewak = DDAEWAK(autoenc,X |> gpu)

L_ddaewak = @showprogress mapreduce(vcat,1:steps) do _
    update!(ddaewak,G_i->loss(ddaewak,G_i),
            window_d,window_k)
end
