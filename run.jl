using SemanticWords
using JSON
using KernelMethods.KMap
using KernelMethods.Scores
using KernelMethods.Kernels
using KernelMethods.Supervised
using SimilaritySearch
using NearNeighborGraph
using TextModel

function read_dense(filename; header=true, delim=' ')
    words = String[]
    vectors = DenseCosine{Float32}[]

    for (lineno, line) in enumerate(readlines(filename))
        if header && lineno == 1
            continue
        end
        arr = split(line, delim)
        m = length(arr)
        vec = Vector{Float32}(m - 1)
        @inbounds for i in 2:m
            num = parse(Float32, arr[i])
            isnan(num) && error("parsed NaN object in $filename, lineno: $lineno -- $line")
            isinf(num) && error("parsed Inf object in $filename, lineno: $lineno -- $line")
            vec[i-1] = num
        end
        
        try
            v = DenseCosine(vec)
            push!(vectors, v)
        catch
            info("WARNING ignoring vector with zero norm; filename: $filename, lineno: $lineno")
            continue
        end
        push!(words, arr[1])

        if (length(vectors) % 50000) == 1
            info("advance $lineno -- #vectors: $(length(vectors))")
        end
    end
    
    info("a vocabulary of $(length(words)) has loaded")
    words, vectors
end

function read_sparse(filename; wdelim='\t', delim=' ')
    words = String[]
    vectors = VBOW[]

    f = open(filename) 
    for (lineno, line) in enumerate(readlines(filename))
        if lineno == 1
            config = JSON.parse(line)
            continue
        end

        w, line = split(line, wdelim, limit=2)
        push!(words, w)
        arr = split(line, delim)
        m = length(arr)
        vec = Vector{WeightedToken}(m)
        @inbounds for i in 1:m
            a, b = split(arr[i], ':')
            num = parse(Float32, b)
            isnan(num) && error("parsed NaN object in $filename, lineno: $lineno -- $line")
            isinf(num) && error("parsed Inf object in $filename, lineno: $lineno -- $line")
            vec[i] = WeightedToken(parse(Int, a), num)
        end
        
        push!(vectors, VBOW(vec))

        if (length(vectors) % 50000) == 1
            info("advance $lineno -- #vectors: $(length(vectors))")
        end
    end
    close(f)
    
    info("a vocabulary of $(length(words)) has loaded")
    words, vectors
end

function create_config(nlist, qlist, skiplist=[])
    config = TextConfig()
    config.del_usr = false
    config.del_punc = false
    config.del_num = false
    config.del_url = true
    config.del_diac = false
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
            #sort!(codes[i])
            # H[w[1]] = [Char(c+512) for c in codes[i]] |> join
            H[w] = [Char(c+512) for c in codes[i]] |> join
        end
    end
    
    function encode(data)::Vector{String}
        if typeof(data) <: Vector
            L = String[]
            for text in data
                for w in tokenize(text::String, config)
                    ww = get(H, w, "")
                    if length(ww) > 0
                        push!(L, w)
                    end
                end
            end
            
            return L
        else
            return [get(H, w, w) for w in tokenize(data::String, config)]
        end
    end
		
    encode
end

function encode_corpus(encode::Function, corpus, key_text)
    for tweet in corpus
        tweet["original-text"] = tweet[key_text]
        tweet[key_text] = join(encode(tweet[key_text]), ' ')
    end

    corpus
end

function vectorize_collection(model, col, key_klass, key_text, allow_zero_length)
	X = Vector{VBOW}(length(col))
	y = Vector{String}(length(col))
	
	i = 0
	for tweet in col
		vec = vectorize(tweet[key_text], model)
		if allow_zero_length || length(vec) > 0
			i += 1
			X[i] = vec
			y[i] = tweet[key_klass]
		end
	end
	
	if !allow_zero_length
		resize!(X, i)
		resize!(y, i)
	end
	X, y
end

function rocchio_model(model, train, test; key_klass="klass", key_text="text")
    info("vectorizing train")
	X, y = vectorize_collection(model, train, key_klass, key_text, false)
    
    info("computing centroids")
    ŷ = unique(y)
    X̂ = [centroid!(X[y .== label]) for label in ŷ]
    index = Sequential(X̂, angle_distance)
    info("vectorizing test")
	Xtest, ytest = vectorize_collection(model, test, key_klass, key_text, true)

    info("predicting test")
    ypred = String[]
    for (i, x) in enumerate(Xtest)
        res = search(index, x, KnnResult(1))
        push!(ypred, ŷ[first(res).objID])
    end

    _scores = scores(ytest, ypred)
    _scores
end

function knnclassifier(model, train, test; key_klass="klass", key_text="text")
    info("*** starting kclassifier procedure ***")
    X, y = vectorize_collection(model, train, key_klass, key_text, false)   
    info("*** computing kclassifier ***")
    nnc = NearNeighborClassifier(X, y, angle_distance)
    KernelMethods.Supervised.optimize!(nnc, accuracy, folds=3)
    info("vectorizing test")
	Xtest, ytest = vectorize_collection(model, test, key_klass, key_text, true)
    
    ypred = predict(nnc, Xtest)
    _scores = scores(ytest, ypred)
    _scores
end

function kclassifier(model, train, test; key_klass="klass", key_text="text")
    info("*** starting kclassifier procedure ***")
    X, y = vectorize_collection(model, train, key_klass, key_text, false)
    _X = VBOW[]
    _y = Vector{typeof(y[1])}()
    ŷ = unique(y)
    for label in ŷ
        info("*** computing centroids for label '$label' ***")
        cc = ApproxKDCentroids(31, 3, maxiter=10, tol=0.001, recall=0.9)
        centroids, codes, rlist = codebook(cc, X[y .== label])
        append!(_X, centroids)
        for i in 1:length(centroids)
            push!(_y, label)
        end
    end
    
    info("*** computing kclassifier ***")
    nnc = NearNeighborClassifier(_X, _y, angle_distance)
    KernelMethods.Supervised.optimize!(nnc, accuracy, folds=3)
    info("vectorizing test")
	Xtest, ytest = vectorize_collection(model, test, key_klass, key_text, true)
    
    ypred = predict(nnc, Xtest)
    _scores = scores(ytest, ypred)
    _scores
end

function create_codebook()
    numcenters = parse(Int, get(ENV, "numcenters", "10000"))
    k = parse(Int, get(ENV, "k", "7"))
    # kcut = parse(Int, get(ENV, "kcut", "")
    kind = get(ENV, "kind", "random")
    maxiter = parse(Int, get(ENV, "maxiter", "10"))
    tol = parse(Float64, get(ENV, "tol", "0.001"))
    @assert (kind in ("random", "fft", "dnet")) "kind=random|fft|dnet; specifies the algorithm to compute the codebook"
    vecfile = get(ENV, "vectors", "")
    @assert (length(vecfile) > 0) "vectors=file; the embedding file is mandatory"
    modelname = get(ENV, "model", "")
    @assert (length(modelname) > 0) "model=outfile"
    recall = parse(Float64, get(ENV, "recall", "0.9"))
    
    @show numcenters, k, kind, recall, maxiter, tol, vecfile
    cc = ApproxKDCentroids(numcenters, k, maxiter=maxiter, tol=tol, recall=recall)
    #cc = FFTraversal(numcenters, k)
    if get(ENV, "vformat", "dense") == "dense"
        begin
            words, X = read_dense(vecfile, header=false)
            centroids, codes, rlist = codebook(cc, X)
            # centroids = []
            # codes = [rand(1:numcenters, k) for i in 1:length(X)]
            @save modelname words centroids codes
        end
    else
        begin
            words, X = read_sparse(vecfile)
            centroids, codes, rlist = codebook(cc, X)
            @save modelname words centroids codes
        end
    end
end

function main()
    action = get(ENV, "action", "")
    key_klass = get(ENV, "klass", "klass")
    key_text = get(ENV, "text", "text")
    weighting = get(ENV, "weighting", "tfidf")
    @assert (weighting in ["tfidf", "tf", "idf", "freq", "entropy"]) "unknown weighting scheme $weighting"
    smoothing = parse(Int, get(ENV, "smoothing", "3"))
    nlist = [parse(Int, i) for i in split(get(ENV, "nlist", "1"), ',') if length(i) > 0]
    qlist = [parse(Int, i) for i in split(get(ENV, "qlist", "1,3,5"), ',') if length(i) > 0]
    filter_low = parse(Int, get(ENV, "filter_low", "1"))
    filter_high = parse(Float64, get(ENV, "filter_high", "1.0"))
         
                            
    if action == "model"
        create_codebook()
    elseif action == "translate"
        modelname=get(ENV, "model", "")
        @assert (length(modelname) > 0) "argument model=modelfile is mandatory"
        @load modelname words centroids codes
        config = create_config([1], [])
        encode = create_encoder(config, words, centroids, codes)
        info("encoding $ARGS")
        key_text = get(ENV, "text", "text")
        @show modelname, key_text, ARGS
        for arg in ARGS
            outname = replace(modelname, ".jld2", "") * "." * basename(arg)
            f = open(outname, "w")
            for tweet in encode_corpus(encode, [JSON.parse(line) for line in readlines(arg)], key_text)
                println(f, JSON.json(tweet))
            end
            close(f)
        end
    elseif action == "evaluate"
        train = [JSON.parse(line) for line in readlines(ARGS[1])]
        test = [JSON.parse(line) for line in readlines(ARGS[2])]

		config = create_config(nlist, qlist)
        X = [item[key_text] for item in train]
        if weighting == "entropy"
            y = [item[key_klass] for item in train]
            le = LabelEncoder(y)
            model = DistModel(config, X, transform.(le, y))
            model = EntModel(model, smoothing)
        else
            vmodel = VectorModel(config)
            vmodel.filter_low = filter_low
            vmodel.filter_high = filter_high
            TextModel.fit!(vmodel, X)
            info("vocabulary size: ", length(vmodel.W))
            if weighting == "tfidf"
                model = TfidfModel(vmodel)
            elseif weighting == "tf"
                model = TfModel(vmodel)
            elseif weighting == "idf"
                model = IdfModel(vmodel)
            elseif weighting == "freq"
                model = FreqModel(vmodel)
            end
        end
                                
        _scores = rocchio_model(model, train, test, key_klass=key_klass, key_text=key_text)
        #_scores = knnclassifier(model, train, test, key_klass=key_klass, key_text=key_text)
		println(JSON.json(_scores))
    elseif action == "semantic-vocabulary"
        train = [JSON.parse(line) for line in readlines(ARGS[1])]
        
		config = create_config(nlist, qlist)
        X = []
        for item in train
            for text in item[key_text]
                push!(X, text)
            end
        end
        # X = [item[key_text] for item in train]
        if weighting == "entropy"
            y = [item[key_klass] for item in train]
            le = LabelEncoder(y)
            model = DistModel(config, X, transform.(le, y))
            model = EntModel(model, 3)
        else
            vmodel = VectorModel(config)
            vmodel.filter_low = filter_low
            vmodel.filter_high = filter_high
            TextModel.fit!(vmodel, X)
            info("vocabulary size: ", length(vmodel.W))
            if weighting == "tfidf"
                model = TfidfModel(vmodel)
            elseif weighting == "tf"
                model = TfModel(vmodel)
            elseif weighting == "idf"
                model = IdfModel(vmodel)
            elseif weighting == "freq"
                model = FreqModel(vmodel)
            end
        end

        XX = [vectorize(text, model) for text in X]
        # clear X
        tokenmap = id2token(vmodel)
        tX = dtranspose(XX)
        @assert length(model.vmodel.W) == length(tX)
        header = Dict(k => getfield(config, k) for k in fieldnames(config) if k != :normalize)
        header[:normalize] = config.normalize != identity
        println(JSON.json(header))
        for (keyid, tokens) in tX
            #info("word $keyid - $(tokenmap[keyid]): ", [(a.id, a.weight) for a in tokens])
            if length(tokens) >= filter_low
                println(tokenmap[keyid], "\t", join([string(a.id |> Int, ":",a.weight) for a in tokens], ' '))
            end
        end

	elseif action == "index"
		filename = ARGS[1]
		indexname = replace(filename, ".json", "") * ".index.jld2"
		if isfile(indexname)
			info("loading index from $indexname")
			@load indexname train model nns
		else
			train = [JSON.parse(line) for line in readlines(filename)]
			config = create_config(nlist, qlist)
			model = VectorModel(config)
			model.filter_low = filter_low
			model.filter_high = filter_high
			TextModel.fit!(model, train, get_text=x -> x[key_text])
			X, y = vectorize_collection(model, train, key_klass, key_text, false)
			N = LogSatNeighborhood()
			nns = LocalSearchIndex(VBOW, cosine_distance, search=BeamSearch(), neighborhood=N)
			NearNeighborGraph.fit!(nns, X)
			NearNeighborGraph.optimize!(nns, 0.9)
			@save indexname train model nns
		end
		
		for p in search(nns, vectorize("a veces me dan ganas pero luego se me pasa", model), KnnResult(5))
			@show p, train[p.objID][key_klass]
		end
    else
        exit(1)
    end
end

if !isinteractive()
	main()
end
