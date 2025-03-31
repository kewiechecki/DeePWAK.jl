module DeePWAK

using Reexport
using Flux, Functors

using Distributions, MultivariateStats, SparseArrays, Match, OneHotArrays
@reexport using Leiden

#@reexport using PyCall
#import PyCall.set!

@reexport using Autoencoders
import Autoencoders.encode
import Autoencoders.decode

import Autoencoders.diffuse
import Autoencoders.dist
import Autoencoders.kern

import Autoencoders.cluster
import Autoencoders.centroid
import Autoencoders.partition

import Autoencoders.loss

@reexport using TrainingIO
import TrainingIO.update!

export encode, decode, diffuse, dist, kern
export cluster, centroid, partiton
export loss, lossfn

export update!, updateparam!

export AbstractDEWAK, AbstractDeePWAK
export DEWAK, DEPWAK, DDAEWAK
export params, set!, data, pca, knn, graph, losslog
export perm, clusts, partitionmat, modularity

include("clustering.jl")
include("DEWAK.jl")
include("DEPWAK.jl")
include("DDAEWAK.jl")
include("train.jl")

end # module DeePWAK
