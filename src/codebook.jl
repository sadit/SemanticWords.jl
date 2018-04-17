using KernelMethods.KMap: fftraversal, size_criterion, dnet
using SimilaritySearch
using NearNeighborGraph

export codebook

function dnetcodebook(X::AbstractVector{DenseCosine{Float32}}, numcenters, k)
    n = length(X)
    rlist = Int[]
    xcodes = KnnResult[KnnResult(k) for i in 1:n]
    # dmax = 0.0
    function callback(pivID, nearlist)
        push!(rlist, pivID)
        
        push!(xcodes[pivID], pivID, 0.0)        
        for p in nearlist
            push!(xcodes[p.objID], pivID, p.dist)
        end
        # dmax = _dmax
    end

    dnet(callback, X, CosineDistance(), ceil(Int, n / numcenters) |> Int)

    sort!(rlist)
    XX = X[rlist]
    H = Dict(r => i for (i, r) in enumerate(rlist))
    # codes = [[H[first(xcodes[i]).objID]] for i in 1:n]
    codes = [[H[p.objID] for p in xcodes[i]] for i in 1:n]

    info("creating inverted index")
    invindex = [Int[] for i in 1:length(rlist)]
    for (objID, plist) in enumerate(codes)
        for refID in plist
            push!(invindex[refID], objID)
        end
    end

    info("computing centroids")
    centroids = Vector{DenseCosine{Float32}}(length(rlist))  # computes the centroid of each reference
    Threads.@threads for i in 1:length(invindex)
        plist = invindex[i]
        if length(plist) == 0
            centroids[i] = XX[i]
        else
            centroids[i] = centroid(@view X[plist])
        end
    end
    
    centroids, codes, rlist
end


function fftcodebook(X::AbstractVector{DenseCosine{Float32}}, numcenters, k)
    n = length(X)
    rlist = Int[]
    # dmax = 0.0
    function callback(pivID, _dmax)
        push!(rlist, pivID)
        # dmax = _dmax
    end

    xcodes = KnnResult[KnnResult(k) for i in 1:n]
    function callbackdist(pivID, objID, d)
        push!(xcodes[objID], pivID, d)
    end
    fftraversal(callback, X, CosineDistance(), size_criterion(numcenters), callbackdist)
    sort!(rlist)
    centers = X[rlist]
    H = Dict(r => i for (i, r) in enumerate(rlist))
    codes = [[H[p.objID] for p in xcodes[i]] for i in 1:n]
    centers, codes, rlist
end

function create_index(db)
    dist = CosineDistance()
    # I = Sequential(db, dist)
    N = LogSatNeighborhood()
    I = LocalSearchIndex(DenseCosine{Float32}, dist, search=BeamSearch(), neighborhood=N)
    fit!(I, db)
    NearNeighborGraph.optimize!(I, 0.9)
    I
end

function centroid(vecs::AbstractVector{DenseCosine{Float32}})
    m = length(vecs[1].vec)
    w = zeros(Float32, m)
    
    for vv in vecs
        v::Vector{Float32} = vv.vec
        @inbounds @simd for i in 1:m
            w[i] = w[i] + v[i]
        end
    end

    DenseCosine(w)
end

function codebook(X::AbstractVector{DenseCosine{Float32}}, numcenters, k; maxiter=10, tol=0.001)
    n = length(X)
    rlist = rand(1:n, numcenters) |> Set |> collect   # initial selection
    _empty = Int[]
    codes = [_empty for i in 1:n]
    C = X[rlist]
    iter = 0
    distances = zeros(Float64, n)
    scores = [0, typemax(Float64)]
    mutex = Threads.Mutex()
    while iter < maxiter && abs(scores[end-1] - scores[end]) > tol
        iter += 1
        I = create_index(C)
        info("\n******** computing all $k nearest references; iteration: $iter; score: $scores ********\n")
        Threads.@threads for objID in 1:n
            # for objID in 1:n
            x = X[objID]
            res = search(I, x, KnnResult(k))
            codes[objID] = [p.objID for p in res]
            distances[objID] = last(res).dist

            #if (objID % 10_000) == 1
            #    info("advance $(objID) of $(n) [$(round(objID/n * 100, 2))%], using $(numC) C, $(k) nearest references -- $(Dates.now())")
            #end
        end
        
        push!(scores, mean(distances))

        info("******** new score with $k references: $scores ********")
        info("******** creating inverted index")
        invindex = [Int[] for i in 1:numcenters]
        for (objID, plist) in enumerate(codes)
            for refID in plist
                push!(invindex[refID], objID)
            end
        end
        
        info("******** computing centroids")
        # centroids = Vector{DenseCosine{Float32}}(length(rlist))  # computes the centroid of each reference
        
        Threads.@threads for i in 1:length(invindex)
            plist = invindex[i]
            if length(plist) > 0  # C[i] can be empty because we could be using approximate search
                C[i] = centroid(@view X[plist])
            end
        end
    end
    info("******** finished computation of $k references: $scores ********")
    C, codes, rlist
       
end
