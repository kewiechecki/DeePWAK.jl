module DeePWAK

using Reexport
using Flux, Functors

using Distributions, MultivariateStats, SparseArrays
@reexport using Leiden

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
export loss

export update!

export AbstractDEWAK, AbstractDeePWAK
export DEWAK, DEPWAK, DDAEWAK
export data, pca, knn, losslog
export perm, clusts, partitionmat

include("clustering.jl")
include("DEWAK.jl")
include("DEPWAK.jl")
include("DDAEWAK.jl")

end # module DeePWAK
