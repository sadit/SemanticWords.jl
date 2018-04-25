using SemanticWords
using JSON
using KernelMethods.KMap
using KernelMethods.Scores
using KernelMethods.Kernels
using KernelMethods.Supervised
using SimilaritySearch
using TextModel
import KernelMethods.KMap: centroid


function read_vocabulary(filename, binfile)
    if !isfile(binfile)
        words = String[]
        vectors = DenseCosine{Float32}[]
        
        for (lineno, line) in enumerate(readlines(filename))
            if lineno == 1
                continue
            end
            arr = split(line)
            push!(words, arr[1])
            m = length(arr)
            vec = Vector{Float32}(m - 1)
            for i in 2:m
                vec[i-1] = parse(Float32, arr[i])
            end
            push!(vectors, DenseCosine(vec))
            
            if (length(vectors) % 10000) == 1
                info("advance $lineno -- #vectors: $(length(vectors))")
            end
        end

        @save binfile words vectors
    else
        info("loading data from cache file $binfile")
        @load binfile words vectors
    end
    
    info("a vocabulary of $(length(words)) has loaded")
    words, vectors
end


function translate_microtc(encode, filename, outname)
    open(outname, "w") do f
        for line in readlines(filename)
            tweet = JSON.parse(line)
            t = tweet["text"]
            tweet["original-text"] = t
            tweet["text"] = join(encode(t), ' ')
            println(f, JSON.json(tweet))
        end
    end
end

function create_config(nlist, qlist, skiplist=[])
    config = TextConfig()
    config.del_usr = true
    config.del_punc = false
    config.del_num = false
    config.del_url = true
    config.nlist = collect(Int, nlist)
    config.qlist = collect(Int, qlist)
    config.skiplist = collect(Int, skiplist)
    config
end

function create_encoder(config, words, centroids, codes)
    H = Dict{String,String}()
    for i in 1:length(words)
        # w = tokenize(words[i], config)
        w = words[i]
        
        # the encoding is designed to preserve these special chars
        # '!' = Char(33) => missing / unknown token
        # ' ' = Char(32) => space, token separator
        if length(w) > 0
            sort!(codes[i])
            # H[w[1]] = [Char(c+512) for c in codes[i]] |> join
            H[w] = [Char(c+512) for c in codes[i]] |> join
        end
    end
    
	function encode(text::String)
        [get(H, w, w) for w in tokenize(text, config)]
    end
		
    encode
end

function encode_corpus(encode::Function, corpus)
    for tweet in corpus
        tweet["original-text"] = tweet["text"]
        tweet["text"] = join(encode(tweet["text"]), ' ')
    end

    corpus
end

function centroid(vecs::AbstractVector{VBOW})
    u = vecs[1]
    for v in @view vecs[2:end]
        u = u + v
    end

    u
end

function rocchio_model(config, train, test; key_klass="klass", key_text="text")
    model = VectorModel(config)
    model.filter_low = 1
    model.filter_high = 0.9
    fit!(model, train, get_text=x -> x[key_text])
    X = VBOW[]
    y = String[]

    info("vectorizing train")
    @time begin
        for tweet in train
            vec = vectorize(tweet[key_text], model)
            if length(vec) > 0
                push!(X, vec)
                push!(y, tweet[key_klass])
            end
        end
    end

    info("computing centroids")
    ŷ = unique(y)
    X̂ = [centroid(X[y .== label]) for label in ŷ]
    index = Sequential(X̂, angle_distance)
    info("vectorizing test")
    Xtest = VBOW[vectorize(x[key_text], model) for x in test]
    ytest = String[x[key_klass] for x in test]

    info("predicting test")
    ypred = String[]
    for (i, x) in enumerate(Xtest)
        res = search(index, x, KnnResult(1))
        push!(ypred, ŷ[first(res).objID])
    end

    @show accuracy(ytest, ypred), f1(ytest, ypred, weight=:macro)
end

function create_codebook()
    numcenters = parse(Int, get(ENV, "numcenters", "10000"))
    k = parse(Int, get(ENV, "k", "7"))
    # kcut = parse(Int, get(ENV, "kcut", "")
    kind = get(ENV, "kind", "random")
    maxiter = parse(Int, get(ENV, "maxiter", "10"))
    tol = parse(Float64, get(ENV, "tol", "0.001"))
    @assert (kind in ("random", "fft", "dnet")) "kind=random|fft|dnet; specifies the algorithm to compute the codebook"
    vecfile=get(ENV, "vectors", "")
    @assert (length(vecfile) > 0) "vectors=file; the embedding file is mandatory"
    modelname=get(ENV, "model", "")
    @assert (length(modelname) > 0) "model=outfile"
    recall=parse(Float64, get(ENV, "recall", "0.9"))
    
    words, X = read_vocabulary(vecfile, vecfile * ".jld2")
    centroids, codes = compute_cluster(X, numcenters, k, kind, maxiter=maxiter, tol=tol, recall=recall)
    @save modelname words centroids codes
end

if !isinteractive()
    action = get(ENV, "action", "")

    if action == "model"
        create_codebook()
    elseif action == "translate"
        modelname=get(ENV, "model", "")
        @assert (length(modelname) > 0) "argument model=modelfile is mandatory"
        @load modelname words centroids codes
        config = create_config([1], [])
        encode = create_encoder(config, words, centroids, codes)
        info("encoding $ARGS")
        
        for arg in ARGS
            outname = replace(modelname, ".jld2", "") * "." * basename(arg)
            f = open(outname, "w")
            for tweet in encode_corpus(encode, [JSON.parse(line) for line in readlines(arg)])
                println(f, JSON.json(tweet))
            end
            close(f)
        end
    elseif action == "evaluate"
        config = create_config([1], [3, 5])
        train = [JSON.parse(line) for line in readlines(ARGS[1])]
        test = [JSON.parse(line) for line in readlines(ARGS[2])]
		key_klass = get(ENV, "klass", "klass")
		key_text = get(ENV, "text", "text")
        rocchio_model(config, train, test, key_klass=key_klass, key_text=key_text)
    else
        exit(1)
    end

    
    # outname = joinpath(output, basename(arg))
    # outname = outname * ".sem.kind=$s.numcenters=$numcenters.k=$k"
    # info("encoding $arg -> $outname")
    # translate_microtc(encode, arg, outname)
end
