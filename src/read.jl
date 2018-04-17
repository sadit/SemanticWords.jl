using SimilaritySearch
using JLD2

export read_vocabulary

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
