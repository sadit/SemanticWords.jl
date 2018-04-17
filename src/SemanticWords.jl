using JSON
using JLD2

module SemanticWords

import SimilaritySearch: L2Distance, CosineDistance, DenseCosine, Sequential, KnnResult
import NearNeighborGraph: optimize!, fit!, LocalSearchIndex, BeamSearch, FixedNeighborhood, LogNeighborhood, LogSatNeighborhood


include("read.jl")
include("codebook.jl")

export compute_cluster

function compute_cluster(X, numcenters, k, kind; maxiter=10, tol=0.001)
    if kind == "random"
        centroids, codes, rlist = codebook(X, numcenters, k, maxiter=maxiter, tol=tol)
    elseif kind == "fft"
        centroids, codes, rlist = fftcodebook(X, numcenters, k)
    else # kind == "dnet"
        centroids, codes, rlist = dnetcodebook(X, numcenters, k)
    end

    centroids, codes
end


end


