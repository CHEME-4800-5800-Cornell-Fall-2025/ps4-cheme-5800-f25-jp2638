function _safe_log(x::Float64)
    if x <= 0.0
        return -1.0e10; # a large negative number
    else
        return log(x);
    end
end

function _objective_function(w::Array{Float64,1}, ḡ::Array{Float64,1}, 
    Σ̂::Array{Float64,2}, R::Float64, μ::Float64, ρ::Float64)

    # Simplified version without barrier term since we enforce non-negativity directly
    f = w'*(Σ̂*w) + (1/(2*ρ))*((sum(w) - 1.0)^2 + (transpose(ḡ)*w - R)^2);

    return f;
end

"""
    function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
        verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
        α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
        μ::Float64 = 1.0, ρ::Float64 = 1.0) -> MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem

The `solve` function solves the minimum variance portfolio allocation problem using a simulated annealing approach for a given instance 
    of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) problem type.

### Arguments
- `model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: An instance of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) that defines the problem parameters.
- `verbose::Bool = true`: A boolean flag to control verbosity of output during optimization.
- `K::Int = 10000`: The initial number of iterations at each temperature level.
- `T₀::Float64 = 1.0`: The initial temperature for the simulated annealing process.
- `T₁::Float64 = 0.1`: The final temperature for the simulated annealing process.
- `α::Float64 = 0.99`: The cooling rate for the temperature.
- `β::Float64 = 0.01`: The step size for generating new candidate solutions.
- `τ::Float64 = 0.99`: The penalty parameter update factor.
- `μ::Float64 = 1.0`: The initial penalty parameter for the logarithmic barrier term.
- `ρ::Float64 = 1.0`: The initial penalty parameter for the equality constraints.

### Returns
- `MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: The input model instance updated with the optimal portfolio weights.

"""
function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
    verbose::Bool = true, K::Int = 100000, T₀::Float64 = 1000.0, T₁::Float64 = 1e-8, 
    α::Float64 = 0.99, β::Float64 = 0.02, τ::Float64 = 0.99,
    μ::Float64 = 1000.0, ρ::Float64 = 1000.0)

    # initialize -
    has_converged = false;

    # unpack the model parameters -
    w = model.w;
    ḡ = model.ḡ;
    Σ̂ = model.Σ̂;
    R = model.R;

    # Pre-allocate memory for matrix operations
    Σ̂w = zeros(length(w))
    
    # Initialize simulated annealing temperature
    T = T₀; # Initial temperature controls acceptance probability of worse solutions
    
    # Initialize portfolio weights
    current_w = copy(w);
    current_w .= abs.(current_w); # Ensure non-negative weights for valid portfolio
    
    # Initialize tracking variables for initial solution search
    best_init_error = Inf    # Track smallest return target error found
    best_init_w = copy(current_w)  # Store best initial weights
    
    # Smart initialization: Try 1000 random portfolios to find good starting point
    for _ in 1:1000
        # Generate random portfolio with non-negative weights
        tmp_w = abs.(randn(length(w)))
        tmp_w ./= sum(tmp_w)  # Normalize to satisfy sum-to-one constraint
        
        # Check how well this portfolio meets target return
        ret_error = abs(dot(ḡ, tmp_w) - R)  # Error from desired return
        
        # Keep track of best initialization found
        if ret_error < best_init_error
            best_init_error = ret_error
            best_init_w .= tmp_w
            
            # Early stop if we found a very good initial solution
            if ret_error < 0.01  # Within 1% of target return
                break
            end
        end
    end
    
    # Use the best initial solution found
    current_w .= best_init_w
    
    # Pre-compute initial objective value with adaptive penalties
    mul!(Σ̂w, Σ̂, current_w)
    ret_penalty = (dot(ḡ, current_w) - R)^2
    sum_penalty = (sum(current_w) - 1.0)^2
    current_f = dot(current_w, Σ̂w) + (1/ρ)*(sum_penalty + 100.0*ret_penalty);
    
    # Best solution tracking with strict feasibility checks
    w_best = copy(current_w);
    f_best = current_f;
    best_ret_penalty = ret_penalty;
    best_sum_penalty = sum_penalty;
    
    # Adaptive iteration control
    KL = K;
    no_improvement_count = 0;

    while has_converged == false
    
        accepted_counter = 0; 
        
        # Simulated annealing main loop
        for _ in 1:KL
            # Generate a new candidate solution
            candidate_w = copy(current_w)
            
            # More sophisticated perturbation strategy
            i, j = rand(1:length(candidate_w), 2)
            
            # Adaptive step size based on temperature
            δ = β * T * candidate_w[i]
            
            # Ensure non-negativity constraint
            if candidate_w[i] - δ >= 0
                candidate_w[i] -= δ
                candidate_w[j] += δ
                
                # Normalize to maintain sum-to-one constraint
                candidate_w ./= sum(candidate_w)
            end
            
            # Calculate portfolio metrics and constraint violations
            mul!(Σ̂w, Σ̂, candidate_w)  # Efficient matrix multiplication for risk calculation
            ret_error = abs(dot(ḡ, candidate_w) - R)  # How far from target return
            ret_penalty = ret_error^2  # Squared penalty for return constraint
            sum_penalty = (sum(candidate_w) - 1.0)^2  # Penalty for sum-to-one constraint
            risk_term = dot(candidate_w, Σ̂w)  # Portfolio variance (risk measure)
            
            # Adaptive penalty weights that increase with constraint violation
            ret_weight = 1000.0 * (1.0 + 100.0 * (ret_error > 0.01))  # Higher penalty for >1% return error
            temp_factor = (T₀/T)^0.5  # Increase constraint importance as we cool
            
            # Combined objective: risk + temperature-scaled constraint penalties
            candidate_f = risk_term + temp_factor*(sum_penalty/ρ + ret_weight*ret_penalty)
            
            # Calculate change in objective function
            Δf = candidate_f - current_f
            
            # Calculate base acceptance probability using Metropolis criterion
            acceptance_prob = exp(-Δf/T)  # Standard simulated annealing acceptance
            
            # Increase acceptance probability for solutions closer to target return
            if ret_error < best_ret_penalty^0.5  # If error is improving
                acceptance_prob *= 10.0  # Make acceptance 10x more likely
            end
            
            # Strongly favor solutions that satisfy all constraints
            if ret_error < 0.01 && sum_penalty < 1e-4  # Within 1% of target & sum ≈ 1
                acceptance_prob *= 10.0  # Further increase acceptance chance
            end
            
            # Reduce probability of accepting solutions that worsen return significantly
            if ret_error > 2.0 * best_ret_penalty^0.5  # If error more than doubles
                acceptance_prob *= 0.1  # Make acceptance 10x less likely
            end
            
            if Δf <= 0 || rand() < acceptance_prob
                current_w = copy(candidate_w)
                current_f = candidate_f
                accepted_counter += 1
                
                # Update best solution if current is better and closer to target return
                # Update best solution with strict feasibility checks
                is_better_feasible = (ret_penalty < 1e-4 && sum_penalty < 1e-4 && 
                                    (current_f < f_best || best_ret_penalty > 1e-4))
                is_better_infeasible = (current_f < f_best && 
                                      ret_penalty < best_ret_penalty && 
                                      sum_penalty < best_sum_penalty)
                
                if is_better_feasible || (is_better_infeasible && !any(w_best .< -1e-10))
                    w_best = copy(current_w)
                    f_best = current_f
                    best_ret_penalty = ret_penalty
                    best_sum_penalty = sum_penalty
                    no_improvement_count = 0  # Reset counter on improvement
                else
                    no_improvement_count += 1
                end
            end
        end

        # update KL -
        fraction_accepted = accepted_counter/KL; # what is the fraction of accepted moves
        
        # Adaptively adjust iteration count based on acceptance rate
        if (fraction_accepted > 0.8)
            # Too many acceptances - decrease iterations but ensure minimum
            KL = max(50, ceil(Int, 0.9*KL));
        elseif (fraction_accepted < 0.2)
            # Too few acceptances - increase iterations but cap maximum
            KL = min(K, ceil(Int, 1.1*KL));
        end

        # Adaptive cooling schedule based on constraint satisfaction
        if best_ret_penalty > 1e-4  # If target return not met
            T *= α^0.5  # Cool more slowly to explore more solutions
            ρ *= 0.95  # Gradually increase penalty weight for constraints
        else  # Good progress with return constraint
            T *= α  # Standard cooling rate
            ρ *= τ  # Standard penalty update
        end
        
        # Multi-criteria convergence check
        if T ≤ T₁ || (best_ret_penalty < 1e-4 && best_sum_penalty < 1e-4 && no_improvement_count > 5000)
            # Check solution quality before accepting convergence
            if best_ret_penalty < 1e-2 && best_sum_penalty < 1e-4  # Good solution found
                has_converged = true
            elseif T ≤ T₁/10  # Temperature very low but solution not ideal
                has_converged = true  # Stop anyway to avoid wasting computation
            end
        end
    end

    # update the model with the optimal weights -
    model.w = w_best;

    # return the model -
    return model;
end

"""
    function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem) -> Dict{String,Any}

The `solve` function solves the Markowitz risky asset-only portfolio choice problem for a given instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) problem type.
The `solve` method checks for the optimization's status using an assertion. Thus, the optimization must be successful for the function to return.
Wrap the function call in a `try` block to handle exceptions.


### Arguments
- `problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`: An instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) that defines the problem parameters.

### Returns
- `Dict{String, Any}`: A dictionary with optimization results.

The results dictionary has the following keys:
- `"reward"`: The reward associated with the optimal portfolio.
- `"argmax"`: The optimal portfolio weights.
- `"objective_value"`: The value of the objective function at the optimal solution.
- `"status"`: The status of the optimization.
"""
function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem)::Dict{String,Any}

    # initialize -
    results = Dict{String,Any}()
    Σ = problem.Σ;
    μ = problem.μ;
    R = problem.R;
    bounds = problem.bounds;
    wₒ = problem.initial

    # setup the problem -
    d = length(μ)
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
    @variable(model, bounds[i,1] <= w[i=1:d] <= bounds[i,2], start=wₒ[i])

    # set objective function -
    @objective(model, Min, transpose(w)*Σ*w);

    # setup the constraints -
    @constraints(model, 
        begin
            # my turn constraint
            transpose(μ)*w >= R
            sum(w) == 1.0
        end
    );

    # run the optimization -
    optimize!(model)

    # check: was the optimization successful?
    @assert is_solved_and_feasible(model)

    # populate -
    w_opt = value.(w);
    results["argmax"] = w_opt
    results["reward"] = transpose(μ)*w_opt; 
    results["objective_value"] = objective_value(model);
    results["status"] = termination_status(model);

    # return -
    return results
end