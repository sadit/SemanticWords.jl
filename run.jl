using SemanticWords
using JSON
using KernelMethods.KMap
using KernelMethods.Scores
using KernelMethods.Kernels
using KernelMethods.Supervised
using SimilaritySearch
using TextModel
import KernelMethods.KMap: centroid

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

function create_config()
    config = TextConfig()
    config.del_usr = true
    config.del_punc = false
    config.del_num = false
    config.del_url = true
    config.nlist = []
    config.qlist = [1,3]
    config.skiplist = []

    config
end

function create_encoder(words, centroids, codes)
    H = Dict{String,String}()
    config = create_config()
    
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

function rocchio_model(train, test)
    config = create_config()
    model = VectorModel(config)
    model.filter_low = 10
    model.filter_high = 0.9
    fit!(model, train, get_text=x -> x["text"])
    X = VBOW[]
    y = String[]

    info("vectorizing train")
    @time begin
        for tweet in train
            vec = vectorize(tweet["text"], model)
            if length(vec) > 0
                push!(X, vec)
                push!(y, tweet["klass"])
            end
        end
    end

    info("XXXXXXXXXXX KernelClassifier XXXXXXXXX")
    #classifier = KernelClassifier(X, y, ensemble_size=1, space=KConfigurationSpace(distances=[() -> cosine_distance],
    #                                                                               classifiers=[NearNeighborClassifier],
    #                                                                               kernels=[linear_kernel, gaussian_kernel],
    #                                                                               sampling=[(dnet, 30), (dnet, 100)]
    #                                                                               ))
    info("computing centroids")
    ŷ = unique(y)
    X̂ = [centroid(X[y .== label]) for label in ŷ]
    index = Sequential(X̂, angle_distance)
    info("vectorizing test")
    Xtest = VBOW[vectorize(x["text"], model) for x in test]
    ytest = String[x["klass"] for x in test]

    info("predicting test")
    ypred = String[]
    # ypred2 = String[]
    for (i, x) in enumerate(Xtest)
        res = search(index, x, KnnResult(1))
        push!(ypred, ŷ[first(res).objID])
    end

    #ypred2 = predict(classifier, Xtest)
    @show accuracy(ytest, ypred), f1(ytest, ypred, weight=:macro)
    # @show accuracy(ytest, ypred2), f1(ytest, ypred2, weight=:macro)
end

function search_model(encode, corpus)
    y = [tweet["klass"] for tweet in corpus]
    config = create_config()
    model = VectorModel(config)
    fit!(model, corpus, get_text=x -> x["text"])
    X = [vectorize(tweet["text"], model) for tweet in corpus]
    classifier = KernelClassifier(X, y, ensemble_size=5, space=KConfigurationSpace(distances=[() -> angle_distance],
                                                                                   classifiers=[NearNeighborClassifier],
                                                                                   kernels=[linear_kernel, gaussian_kernel],
                                                                                   sampling=[(dnet, 30), (dnet, 100)]
                                                                                   ))
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

    if kind == "random"
        s = "random.maxiter=$maxiter.tol=$tol"
    else
        s = kind
    end
    output = "model.$(basename(vecfile)).kind=$s.centers=$numcenters.k=$k"

    if !isdir(output)
        mkdir(output)
    end
    
    clustername = joinpath(output, "cluster.jld2")
    if !isfile(clustername)
        # @load clustername words centroids codes
        # else
        words, X = read_vocabulary(vecfile, vecfile * ".jld2")
        centroids, codes = compute_cluster(X, numcenters, k, kind, maxiter=maxiter, tol=tol)
        @save clustername words centroids codes
    end
end


function show_help()
    println("""

usage: arg1=val1 arg2=val2 ... julia run.jl inputfiles...

arguments:

action=train|codebook

For codebook:
numcenters=medium to large integer
k=small integer
maxiter=maybe a small integer
tol=a small floating point (tolerance to conversion)
kind=random|fft|dnet
vectors=embedding file

""")
    
end

if !isinteractive()
    action = get(ENV, "action", "help")
    if action == "help"
        show_help()
    elseif action == "codebook"
        create_codebook()
    elseif action == "train"
        modelname=get(ENV, "model", "")
        @assert (length(modelname) > 0) "argument model=modelfile is mandatory"
        @load modelname words centroids codes
        encode = create_encoder(words, centroids, codes)
        info("encoding $ARGS")
        train = encode_corpus(encode, [JSON.parse(line) for line in readlines(ARGS[1])])
        test = encode_corpus(encode, [JSON.parse(line) for line in readlines(ARGS[2])])

        #train = [JSON.parse(line) for line in readlines(ARGS[1])]
        #test = [JSON.parse(line) for line in readlines(ARGS[2])]
        # search_model(encode, [JSON.parse(line) for line in readlines(arg)])
        info("training on $modelname")
        rocchio_model(train, test)
    else
        show_help()
        exit(1)
    end

    
    # outname = joinpath(output, basename(arg))
    # outname = outname * ".sem.kind=$s.numcenters=$numcenters.k=$k"
    # info("encoding $arg -> $outname")
    # translate_microtc(encode, arg, outname)
end
