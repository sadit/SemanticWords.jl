using SemanticWords
using JSON
using KernelMethods.KMap
using KernelMethods.Scores
using KernelMethods.Kernels
using KernelMethods.Supervised
using SimilaritySearch
using NearNeighborGraph
using TextModel

function parse_vector(vstring, fdelim=',')
    arr = split(vstring, fdelim)
    m = length(arr)
    vec = Vector{Float32}(m)
    @inbounds for i in 1:m
        num = parse(Float32, arr[i])
        isnan(num) && error("parsed NaN object $(arr[i]) -- $vstring")
        isinf(num) && error("parsed Inf object $(arr[i]) -- $vstring")
        vec[i] = num
    end
    
    DenseCosine(vec)
end

function parse_vector_list(text, vdelim=' ')
    h = DenseCosine{Float32}[]
    for vstring in split(strip(text), vdelim)
        try
            v = parse_vector(vstring)
            push!(h, v)
        catch
            info("*** ignoring:", vstring)
            continue
        end   
    end
    
    h
end

function read_vectors(filename, key_vectors)
    f = open(filename)
    db = DenseCosine{Float32}[]
    words = String[]
    
    for (i, line) in enumerate(eachline(f))
        v = JSON.parse(line)
        vecs = parse_vector_list(v[key_vectors])
        append!(db, vecs)
        if i % 50000 == 1
            info("read $i vectors from $filename")
        end
    end
    close(f)
    
    [string(i) for i in 1:length(db)], db
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


function create_encoder(centroids, k, recall)
    I = SemanticWords.create_index(centroids, recall)
    
    function encode(vlist::String)::String
        X = Char[]
        for v in parse_vector_list(vlist)
            res = search(I, v, KnnResult(k))
            for p in res
                push!(X, Char(p.objID+33))
            end
            push!(X, ' ')
        end
        
        X |> join
    end
		
    encode
end


function encode_corpus(encode::Function, corpus, key_text)
    for tweet in corpus
        tweet["original-text"] = tweet[key_text]
        tweet[key_text] = encode(tweet[key_text])  #join(encode(tweet[key_text]), ' ')
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
    
    if get(ENV, "vformat", "dense") == "dense"
        begin
            words, X = read_vectors(vecfile, "text")
            info(length(words), "--", length(X))
            centroids, codes, rlist = codebook(cc, X)
            @save modelname words centroids codes
        end
    else
        error("run-images.jl works only with dense vectors")
        # begin
        #     words, X = read_sparse(vecfile)
        #     centroids, codes, rlist = codebook(cc, X)
        #     @save modelname words centroids codes
        # end
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
    recall = parse(Float64, get(ENV, "recall", "0.9"))
    k = parse(Int, get(ENV, "k", "7"))
                            
    if action == "model"
        create_codebook()
    elseif action == "translate"
        modelname = get(ENV, "model", "")
        @assert (length(modelname) > 0) "argument model=modelfile is mandatory"
        @load modelname words centroids codes
        encode = create_encoder(centroids, k, recall)
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
		println(JSON.json(_scores))
    else
        exit(1)
    end
end

if !isinteractive()
	main()
end
