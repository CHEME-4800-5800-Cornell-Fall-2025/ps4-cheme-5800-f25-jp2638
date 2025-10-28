"""
    function build(modeltype::Type{MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem}, 
        data::NamedTuple)::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem

The `build` function constructs a [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) model from the provided data.

### Arguments
- `modeltype::Type{MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem}`: The type of the model to be constructed.
- `data::NamedTuple`: A named tuple containing the problem data.

The `data` named tuple should contain the following fields:
- `w::Vector{Float64}`: The initial asset weights.
- `R::Float64`: The target return for the portfolio.
- `ḡ::Vector{Float64}`: The expected returns vector.
- `Σ̂::Matrix{Float64}`: The covariance matrix of asset returns.

### Returns
- `MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: The constructed model instance.
"""
## for save purpose
function build(modeltype::Type{MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem}, 
    data::NamedTuple)::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem

    # get stuff from data -
    w = data.w;
    R = data.R;
    ḡ = data.ḡ;
    Σ̂ = data.Σ̂;


    model = modeltype();
    model.w = w;
    model.R = R;
    model.ḡ = ḡ;
    model.Σ̂ = Σ̂;

    return model;
end

"""
    function build(modeltype::Type{MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem}, 
        data::NamedTuple)::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem

The `build` function constructs a [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) model from the provided data.

### Arguments
- `modeltype::Type{MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem}`: The type of the model to be constructed.
- `data::NamedTuple`: A named tuple containing the problem data.

The `data` named tuple should contain the following fields:
- `Σ::Matrix{Float64}`: The covariance matrix of asset returns.
- `μ::Vector{Float64}`: The expected returns vector.
- `bounds::Vector{Tuple{Float64, Float64}}`: The bounds for each asset weight.
- `R::Float64`: The target return for the portfolio.
- `initial::Vector{Float64}`: The initial guess for the asset weights.

### Returns
- `MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`: The constructed model instance.
"""
function build(modeltype::Type{MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem}, 
    data::NamedTuple)::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem

    # get stuff from data -
    Σ = data.Σ;
    μ = data.μ;
    bounds = data.bounds;
    R = data.R;
    initial = data.initial;

    model = modeltype();
    model.Σ = Σ;
    model.μ = μ;
    model.bounds = bounds;
    model.R = R;
    model.initial = initial;

    return model;
end