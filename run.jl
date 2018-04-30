using SemanticWords
using JSON
using KernelMethods.KMap
using KernelMethods.Scores
using KernelMethods.Kernels
using KernelMethods.Supervised
using SimilaritySearch
using NearNeighborGraph
using TextModel
import KernelMethods.KMap: centroid


#=
s = "aaa  0 11 v2
sp = 0; while sp < length(s)
       ep = findnext(s, ' ', sp+1)
       if ep == 0
          ep = length(s)+1
       end
       @show sp, ep, s[sp+1:ep-1]
       sp = ep
       end
=#

function read_vocabulary(filename, binfile; header=true, delim=' ')
    if !isfile(binfile)
        words = String[]
        vectors = DenseCosine{Float32}[]
        
        for (lineno, line) in enumerate(readlines(filename))
            if lineno == 1
                continue
            end
            arr = split(line, delim)
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
    config.del_usr = false
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
    
	#function encode(text::String)
    function encode(data)::Vector{String}
        if typeof(data) <: Vector
            L = String[]
            for text in data
                for w in tokenize(text::String, config)
                    push!(L, get(H, w, w))
                end
            end
            return L
        else
            return [get(H, w, w) for w in tokenize(data::String, config)]
        end
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

"""
Computes a centroid-like sparse vector (i.e., a center under the angle distance) for a collection of sparse vectors.
The computation destroys input array to reduce memory allocations.
"""
function centroid!(vecs::AbstractVector{VBOW})
	lastpos = length(vecs)
	while lastpos > 1
		pos = 1
		for i in 1:2:lastpos
			if i < lastpos
				vecs[pos] = vecs[i] + vecs[i+1]
			end
			pos += 1
		end
		lastpos = pos - 1
	end
	
    vecs[1]
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
	# train = rand(train, 300)
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
	_scores[:accuracy] = accuracy(ytest, ypred)
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
    vecfile=get(ENV, "vectors", "")
    @assert (length(vecfile) > 0) "vectors=file; the embedding file is mandatory"
    modelname=get(ENV, "model", "")
    @assert (length(modelname) > 0) "model=outfile"
    recall=parse(Float64, get(ENV, "recall", "0.9"))
    
    words, X = read_vocabulary(vecfile, vecfile * ".jld2")
    centroids, codes = compute_cluster(X, numcenters, k, kind, maxiter=maxiter, tol=tol, recall=recall)
    @save modelname words centroids codes
end

function main()
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
        train = [JSON.parse(line) for line in readlines(ARGS[1])]
        test = [JSON.parse(line) for line in readlines(ARGS[2])]
		key_klass = get(ENV, "klass", "klass")
		key_text = get(ENV, "text", "text")
        weighting = get(ENV, "weighting", "tfidf")
        @assert (weighting in ["tfidf", "tf", "idf", "freq", "entropy"]) "unknown weighting scheme $weighting"
        nlist = [parse(Int, i) for i in split(get(ENV, "nlist", "1"), ',') if length(i) > 0]
        qlist = [parse(Int, i) for i in split(get(ENV, "qlist", "1,3,5"), ',') if length(i) > 0]
		config = create_config(nlist, qlist)
        X = [item[key_text] for item in train]
        if weighting == "entropy"
            y = [item[key_klass] for item in train]
            le = LabelEncoder(y)
            model = DistModel(config, X, transform.(le, y))
            model = EntModel(model, 3)
        else
            vmodel = VectorModel(config)
            vmodel.filter_low = 2
            vmodel.filter_high = 1.0
            TextModel.fit!(vmodel, X)
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
	elseif action == "index"
		filename = ARGS[1]
		indexname = replace(filename, ".json", "") * ".index.jld2"
		key_klass = get(ENV, "klass", "klass")
		key_text = get(ENV, "text", "text")
		if isfile(indexname)
			info("loading index from $indexname")
			@load indexname train model nns
		else
			train = [JSON.parse(line) for line in readlines(filename)]
			config = create_config([1], [4])
			model = VectorModel(config)
			model.filter_low = 1
			model.filter_high = 1.0
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