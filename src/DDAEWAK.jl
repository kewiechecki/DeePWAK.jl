@doc raw"""
`DDAEWAK(autoencoder :: AbstractEncoder,
         dewak :: DEWAK) <: AbstractDDAE`

See also: `DEWAK`, `AbstractEncoder`, `AbstractDDAE`.
"""
struct DDAEWAK <: AbstractDeePWAK
    autoencoder :: AbstractEncoder
    dewak :: AbstractDEWAK
end
@functor DDAEWAK

function data(M::DDAEWAK)
    M.dewak.dat
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

function kern(M::DDAEWAK)
    kern(M.dewak)
end
