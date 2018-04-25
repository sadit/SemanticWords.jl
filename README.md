# SemanticWords.jl
Provides simple text-based classifiers with semantic approaches

KernelMethods.jl  Languages.jl  NearNeighborGraph.jl  SemanticWords.jl  SimilaritySearch.jl  SnowballStemmer.jl  TextModel.jl

# HOLA

## RAW - Example

Please read `run.jl` to see how it words. A better README.md will be available soon!.


action=train model=model.glove.twitter.27B.25d.txt.kind\=random.maxiter=10.tol=0.001.centers=30000.k=7/cluster.jld2 JULIA_LOAD_PATH=. julia SemanticWords.jl/run.jl semeval2017_english_train.json semeval2017_english_test.json

