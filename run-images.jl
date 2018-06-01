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
    
    function encode(vlist::String)::String
        X = Char[]
        for v in parse_vector_list(vlist)
            res = search(centroids, v, KnnResult(k))
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
        # tweet["original-text"] = tweet[key_text]
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
			y[i] = get(tweet, key_klass, "")
		end
	end
	
	if !allow_zero_length
		resize!(X, i)
		resize!(y, i)
	end
	X, y
end


function rocchio_model(rocchioname, model, gettrain, test; key_klass="klass", key_text="text")
    if isfile(rocchioname)
        @load rocchioname X̂ ŷ
    else
        info("vectorizing train")
        X, y = vectorize_collection(model, gettrain(), key_klass, key_text, false)

        info("computing centroids")
        ŷ = unique(y)
        X̂ = [centroid!(X[y .== label]) for label in ŷ]
        @save rocchioname X̂ ŷ
    end
    
    index = Sequential(X̂, angle_distance)
    info("vectorizing test")
	Xtest, ytest = vectorize_collection(model, test, key_klass, key_text, true)

    info("predicting test")
    ypred = String[]
    Y = []
    for (i, x) in enumerate(Xtest)
        res = search(index, x, KnnResult(length(ŷ)))
        label = ŷ[first(res).objID]
        push!(ypred, label)
        m = test[i]
        delete!(m, "text")
        m[key_klass] = label
        A = [(p.objID, p.dist) for p in res]
        sort!(A)
        m["decision_function"] = [a[2] for a in A]
        push!(Y, m)
    end

    _scores = scores(ytest, ypred)
    Y, _scores
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
    
    info("creating codebook: numcenters:$numcenters, k:$k, kind:$kind, recall=$recall, maxiter=$maxiter, tol:$tol, vecfile=$vecfile")
    cc = ApproxKDCentroids(numcenters, k, maxiter=maxiter, tol=tol, recall=recall)
    
    if get(ENV, "vformat", "dense") == "dense"
        begin
            words, X = read_vectors(vecfile, "text")
            info(length(words), "--", length(X))
            centroids, codes, rlist = codebook(cc, X)
            centroids = SemanticWords.create_index(centroids, recall)
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

        key_text = get(ENV, "text", "text")
        info("*** translating $ARGS with model $modelname")
        for arg in ARGS
            outname = replace(modelname, ".jld2", "") * "." * basename(arg)
            if !isfile(outname)
                f = open(outname, "w")
                for tweet in encode_corpus(encode, [JSON.parse(line) for line in readlines(arg)], key_text)
                    println(f, JSON.json(tweet))
                end
                close(f)
            end
        end
    elseif action == "evaluate"
        trainfile, testfile = ARGS[1], ARGS[2]
        train = []
        function gettrain()
            if length(train) == 0
                for line in readlines(trainfile)
                    push!(train, JSON.parse(line))
                end
            end
                                    
            train
        end
        test = [JSON.parse(line) for line in readlines(testfile)]

		config = create_config(nlist, qlist)
        
        _file = trainfile * ".weighting=$(weighting).filter_low=$(filter_low).filter_high=$(filter_high).nlist=$(join(map(string, nlist), ',')).qlist=$(join(map(string, qlist), ','))"
        rocchiofile = _file * ".rocchio.jld2"
        vmodelfile = _file * ".vmodel.jld2"
                                
        if weighting == "entropy"
            X = [item[key_text] for item in gettrain()]
            y = [item[key_klass] for item in gettrain()]
            le = LabelEncoder(y)
            model = DistModel(config, X, transform.(le, y))
            model = EntModel(model, smoothing)
        else
            if isfile(vmodelfile)
                @load vmodelfile vmodel
            else
                vmodel = VectorModel(config)
                vmodel.filter_low = filter_low
                vmodel.filter_high = filter_high
                X = [item[key_text] for item in gettrain()]
                TextModel.fit!(vmodel, X)
                @save vmodelfile vmodel
            end
            
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
        
        Y, _scores = rocchio_model(rocchiofile, model, gettrain, test, key_klass=key_klass, key_text=key_text)
        println(JSON.json(_scores))
        f = open(testfile * ".predicted", "w")
        for y in Y
            println(f, JSON.json(y))
        end
        close(f)
    else
        exit(1)
    end
end

if !isinteractive()
	main()
end
