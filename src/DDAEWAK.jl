@doc raw"""
`DDAEWAK(autoencoder :: AbstractEncoder,
         dewak :: DEWAK, 
         epoch :: Integer,
         dat :: AbstractArray,
         cache :: DEWAKache) <: AbstractDDAE`

`dewak` should accept `typeof(encode(autoencoder, dat))`.

See also: `DEWAK`, `AbstractEncoder`, `AbstractDDAE`.
"""
struct DDAEWAK <: AbstractDeePWAK
    autoencoder :: AbstractEncoder
    dewak :: AbstractDEWAK
    epoch :: Integer
    dat 
    cache :: DEWAKache
end
@functor DDAEWAK

function DDAEWAK(autoencoder::AbstractEncoder,
                 dewak::AbstractDEWAK,
                 X::AbstractArray; losslabs=[:mse])
    id_params = vcat((names ∘ params)(dewak),"epoch")
    n_params = length(id_params)
    params_0 = DataFrame(Array{Float64}(undef,0,n_params),id_params)
    n_loss = length(losslabs)
    L_0 = DataFrame(Array{Float64}(undef,0,n_loss),
                    losslabs)

    cache = DEWAKache(Dict(), params_0, L_0)
    DDAEWAK(autoencoder, dewak, 0, X, cache)
end

function DDAEWAK(autoencoder::AbstractEncoder,
                 X::AbstractArray;
                 metric=inveucl,
                 losslabs=[:mse])
    E = encode(autoencoder,X) |> cpu
    dewak = DEWAK(E; metric=metric, losslabs=losslabs)
    DDAEWAK(autoencoder, dewak, X; losslabs=losslabs)
end

function params(M::DDAEWAK)
    params(M.dewak)
end

function set!(M::DDAEWAK,param,val)
    set!(M.dewak,param,val)
end

function data(M::DDAEWAK)
    M.dat
end

function encode(M::DDAEWAK, X)
    encode(M.autoencoder,X)
end

function encode(M::DDAEWAK)
    encode(M,M.dewak.dat)
end

function decode(M::DDAEWAK, X)
    decode(M.autoencoder,X)
end

function decode(M::DDAEWAK)
    decode(M,M.dewak.dat)
end

function pca(M::DDAEWAK,X)
    E = encode(M,X)
    pca(M.dewak,E)
end

function kern(M::DDAEWAK,args...)
    kern(M.dewak,args...)
end

function (M::DDAEWAK)(X)
    E = encode(M,X)
    G = kern(M,E |> cpu) |> gpu
    decode(M,(G*E')')
end

function (M::DDAEWAK)(G,X)
    E = encode(M,X)
    decode(M,(G*E')')
end
