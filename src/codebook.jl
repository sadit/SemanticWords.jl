using KernelMethods.KMap: fftraversal, size_criterion, dnet
import KernelMethods.KMap: centroid!
using SimilaritySearch
using NearNeighborGraph
using TextModel

export codebook, FFTraversal, ApproxKDCentroids

abstract type Clustering end

struct FFTraversal <: Clustering 
    numcenters::Int
    k::Int
end

# Approximate k distance centroids
struct ApproxKDCentroids <: Clustering
    numcenters::Int
    k::Int
    maxiter::Int
    tol::Float64
    recall::Float64
end

ApproxKDCentroids(numcenters, k; maxiter=10, tol=0.001, recall=0.9) = ApproxKDCentroids(numcenters, k, maxiter, tol, recall)

function create_distance(::Type{VBOW})
    #cosine_distance
    angle_distance
end

function create_distance(::Type{DenseCosine{Float32}})
    #CosineDistance()
    AngleDistance()
end

function create_distance(::Type{Vector{F}}) where {F <: AbstractFloat}
    L2Distance()
end

"""
Creates an nearest neighbor index to speedup `codebook`
"""
function create_index(db, recall)
    T = typeof(db[1])
    index = LocalSearchIndex(T, create_distance(T), search=BeamSearch(), neighborhood=LogSatNeighborhood(), recall=recall)
    NearNeighborGraph.fit!(index, db)
    index
end


"""
Creates a sampling of the dataset using the farthest first algorithm
(an approximation of the kcenter problem)

"""
function codebook(algo::FFTraversal, X::AbstractVector{T}) where {T <: Union{DenseCosine{Float32},VBOW}}
    n = length(X)
    rlist = Int[]
    # dmax = 0.0
    function callback(pivID, _dmax)
        push!(rlist, pivID)
        # dmax = _dmax
    end

    xcodes = KnnResult[KnnResult(algo.k) for i in 1:n]
    function callbackdist(pivID, objID, d)
        push!(xcodes[objID], pivID, d)
    end
    
    fftraversal(callback, X, CosineDistance(), size_criterion(algo.numcenters), callbackdist)
    sort!(rlist)

    centers = X[rlist]
    H = Dict(r => i for (i, r) in enumerate(rlist))
    codes = [[H[p.objID] for p in xcodes[i]] for i in 1:n]
    centers, codes, rlist
end


function associate_centroids_and_score(algo, C, X, codes, distances)
    index = create_index(C, algo.recall)
    
    Threads.@threads for objID in 1:length(X)
        res = search(index, X[objID], KnnResult(algo.k))
        codes[objID] = [p.objID for p in res]
        distances[objID] = last(res).dist 
    end
    
    mean(distances)
end

const _empty_array = []
"""
Computes a clustering based on random sampling and an approximate nearest neighbor index, the procedure is iterative like the kmeans algorithm but instead of use the nearest neighbor `codebook` uses `k` nearest neighbors to converge and an approximate nn algorithm.

"""
function codebook(algo::ApproxKDCentroids, X::AbstractVector{T}) where {T <: Union{DenseCosine{Float32},VBOW}}
    n = length(X)
    rlist = rand(1:n, algo.numcenters) |> Set |> collect   # initial selection
    codes = [_empty_array for i in 1:n]
    C = X[rlist]
    iter = 0
    distances = zeros(Float64, n)
    scores = [typemax(Float64), associate_centroids_and_score(algo, C, X, codes, distances)]
    
    while iter < algo.maxiter && abs(scores[end-1] - scores[end]) > algo.tol
        iter += 1
        info("*** starting iteration: $iter; scores: $scores ***")
        invindex = [Int[] for i in 1:algo.numcenters]
        for (objID, plist) in enumerate(codes)
            for refID in plist
                push!(invindex[refID], objID)
            end
        end
        
        info("*** computing centroids ***")
        Threads.@threads for i in 1:length(invindex)
            plist = invindex[i]
            # C[i] can be empty because we could be using approximate search
            if length(plist) > 0
                C[i] = centroid!(X[plist])
            end
        end
        
        info("*** computing $(algo.k) nearest references ***")
        s = associate_centroids_and_score(algo, C, X, codes, distances)

        push!(scores, s)
        info("*** new score with $(algo.k) references: $scores ***")
    end
    
    info("*** finished computation of $(algo.k) references, scores: $scores ***")
    C, codes, rlist
end
