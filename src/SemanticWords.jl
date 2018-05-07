using JSON
using JLD2

module SemanticWords
import SimilaritySearch: L2Distance, CosineDistance, DenseCosine, Sequential, KnnResult
import NearNeighborGraph: optimize!, fit!, LocalSearchIndex, BeamSearch, FixedNeighborhood, LogNeighborhood, LogSatNeighborhood


include("codebook.jl")

#export compute_cluster, read_vocabulary


#function compute_cluster(X, numcenters, k, kind; maxiter=10, tol=0.001, recall=0.9)
#    if kind == "random"
#        centroids, codes, rlist = codebook(X, numcenters, k, maxiter=maxiter, tol=tol, recall=recall)
#    elseif kind == "fft"
#        centroids, codes, rlist = fftcodebook(X, numcenters, k)
#    else # kind == "dnet"
#        centroids, codes, rlist = dnetcodebook(X, numcenters, k)
#    end
#
#    centroids, codes
#end

end


