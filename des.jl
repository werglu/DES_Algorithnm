  # differential evolution strategy
  using Distributions
  using DataArrays
  using LinearAlgebra

  function des(par, fn, lower, upper, control)
    @time begin
     function controlParam(name, default)
       if haskey(control, name) == false
           return default
       else
           return control[name]
       end
     end

     function drop(M, r=nothing, c=nothing)
        dr = collect(1:size(M,1))
        dc = collect(1:size(M,2))
        isnothing(r) ? nothing : splice!(dr,r)
        isnothing(c) ? nothing : splice!(dc,c)
        M[dr,dc]
      end

     function sampleFromHistory(history, historySample, lambda)
       ret = []
       for i in 1:lambda
         x = sample(1:size(history[convert(Int64,historySample[i])],2), 1)
         append!(ret, x)
       end
       return ret
     end

     function deleteInfsNaNs(x)
       for i in eachindex(x)
         if isnan(x[i])
           x[i] = prevfloat(typemax(Float64))
         end
       end
       for i in eachindex(x)
         if isinf(x[i])
           x[i] = prevfloat(typemax(Float64))
         end
       end
       return x
     end

     function fn_(x)
       if all(x.>=lower) && all(x.<=upper)
         counteval = counteval + 1
         return fn(x)
       else
         return prevfloat(typemax(Float64))
       end
     end

    function apply(x, f)
      r = []
      for i in 1:size(x, 2)
        append!(r, f(x[:,i]))
      end
      return r
    end

   function fn_l(P)
     if ndims(P) == 2 && size(P, 2) != 1 #if is.matrix()
        if (counteval + size(P, 2)) <= budget
          x = apply(P, fn_)
          return x
        else
          ret = []
          budLeft = budget - counteval
          if budLeft > 0
            for i in 1:budLeft
              append!(ret, fn_(P[:,i]))
            end
          end
          x = append!(ret, repeat([prevfloat(typemax(Float64))], size(P, 2) - budLeft))
          return x
        end
     else
       if counteval < budget
         return fn_(P)
       else
         return prevfloat(typemax(Float64))
       end
     end
   end

   function fn_d(P, P_repaired, fitness)
     P = deleteInfsNaNs(P)
     P_repaired = deleteInfsNaNs(P_repaired)

     if ndims(P) == 2 && ndims(P_repaired) == 2
       repairedInd = []
       for i in 1:size(P,2)
         append!(repairedInd,  all(P[:,i].!= P_repaired[:,i]))
       end
       P_fit = fitness
       P_pom = (P - P_repaired).^2
       vecDist = apply(P_pom,sum)
       if length(vecDist) == 1
         P_fit[findall(repairedInd)] = vecDist[findall(repairedInd)] + worst_fit
       else
         P_fit[findall(repairedInd)] = vecDist[findall(repairedInd)].+ worst_fit
       end
       P_fit = deleteInfsNaNs(P_fit)
       return P_fit
     else
       P_fit = fitness
       if P != P_repaired
        P_pom = (P - P_repaired).^2
         P_fit = worst_fit + sum(P_pom)
         P_fit = deleteInfsNaNs(P_fit)
       end
       return P_fit
     end
   end

   function bounceBackBoundary2(x)
     pom = true
     for i in eachindex(x)
       if x[i] <= lower || x[i] >= upper
         pom = false
       end
     end
     if pom == true
       return x
     elseif any(x.<lower)
       for i in findall(x.<lower)
         x[i] = lower + abs(lower - x[i]) % (upper - lower)
       end
     elseif any(x.>upper)
       for i in findall(x.>upper)
         x[i] = upper - abs(upper - x[i]) % (upper - lower)
       end
     end
     x = deleteInfsNaNs(x)
     return bounceBackBoundary2(x)
   end

   N = length(par)

   # Algorithm parameters
   Ft          = controlParam("Ft", 1) # Scaling factor of difference vectors (a variable!)
   initFt      = controlParam("initFt", 1)
   stopfitness = controlParam("stopfitness", -Inf) # Fitness value after which the convergence is reached
   # Strategy parameter setting:
   budget      = controlParam("budget", 10000 * N) # The maximum number of fitness function calls
   initlambda  = controlParam("lambda", 4 * N) # Population starting size
   lambda      = initlambda ## Population size
   mu          = controlParam("mu", floor(lambda / 2)) # Selection size
   weights     = controlParam("weights", Base.log.(repeat([mu + 1],convert(Int64, mu))) - Base.log.(1:convert(Int64, mu))) # Weights to calculate mean from selected individuals
   weights     = weights / sum(weights) # weights are normalized by the sum
   weightsSumS = sum(weights.^2) # weights sum square
   mueff       = controlParam("mueff", sum(weights)^2 / sum(weights.^2)) # Variance effectiveness factor
   cc          = controlParam("ccum", mu / (mu + 2)) # Evolution Path decay factor
   pathLength  = controlParam("pathLength", 6) # Size of evolution path
   cp          = controlParam("cp", 1 / sqrt(N)) # Evolution Path decay factor
   maxiter     = controlParam("maxit", floor(budget / (lambda + 1))) # Maximum number of iterations after which algorithm stops
   c_Ft        = controlParam("c_Ft", 0)
   pathRatio   = controlParam("pathRatio", sqrt(pathLength)) # Path Length Control reference value
   histSize    = controlParam("history", ceil(6 + ceil(3 * sqrt(N)))) # Size of the window of history - the step length history
   Ft_scale    = controlParam("Ft_scale", ((mueff + 2) / (N + mueff + 3)) / (1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + (mueff + 2) / (N + mueff + 3)))
   tol         = controlParam("tol", 10^-12)
   counteval   = 0 # Number of function evaluations
   sqrt_N      = sqrt(N)

   log_all = controlParam("diag", false)
   log_Ft = controlParam("diag.Ft", log_all)
   log_value = controlParam("diag.value", log_all)
   log_mean = controlParam("diag.mean", log_all)
   log_meanCord = controlParam("diag.meanCords", log_all)
   log_pop = controlParam("diag.pop", log_all)
   log_bestVal = controlParam("diag.bestVal", log_all)
   log_worstVal = controlParam("diag.worstVal", log_all)
   log_eigen = controlParam("diag.eigen", log_all)
   Lamarckism = controlParam("Lamarckism", false)

    best_fit = Inf # The best fitness found so far
    best_par = nothing # The best solution found so far
    worst_fit = nothing # The worst solution found so far
    last_restart = 0
    restart_length = 0
    restart_number = 0

    if log_Ft
      Ft_log = zeros(0, 1)
    end
    if log_value
      value_log = zeros(0, lambda)
    end
    if log_mean
      mean_log = zeros(0, 1)
    end
    if log_meanCord
      meanCords_log = zeros(0, N)
    end
    if log_pop
      pop_log = zeros(N, lambda, maxiter)
    end
    if log_bestVal
      bestVal_log = zeros(0, 1)
    end
    if log_worstVal
      worstVal_log = zeros(0, 1)
    end
    if log_eigen
      eigen_log = zeros(0, N)
    end

    # buffers:
    dMean = zeros(N, convert(Int64,histSize))
    FtHistory = zeros(convert(Int64,histSize)) # Array buffer containing 'histSize' last values of 'Ft'
    pc = zeros(N, convert(Int64,histSize))

    # Initialize internal strategy parameters
    msg = nothing # Reason for terminating
    restart_number = -1
    iter = 0 # Number of iterations

    while counteval < budget

      restart_number = restart_number + 1
      mu = floor(lambda / 2)
      weights = Base.log.(repeat([mu + 1],convert(Int64, mu))) - Base.log.(1:convert(Int64, mu))
      weights = weights / sum(weights)
      weightsPop = Base.log.(repeat([lambda + 1],convert(Int64, lambda))) - Base.log.(1:convert(Int64, lambda))
      weightsPop = weightsPop / sum(weightsPop)

      histHead = 0 # Pointer to the history buffer head
      iter = 0 # Number of iterations
      history = [] # List stores best 'mu'(variable) individuals for 'hsize' recent iterations
      Ft = initFt

      # Create fisrt population
      population = rand(N,lambda).*(0.8*upper-0.8*lower).+0.8*lower

      cumMean = (upper + lower) / 2

      populationRepaired = reshape(apply(population, bounceBackBoundary2), N, lambda)

      if Lamarckism == true
        population = populationRepaired
      end

      selection = fill(0, convert(Int64,mu))
      selectedPoints =  zeros(N, convert(Int64,mu))
      fitness = fn_l(population)

      oldMean = zeros(N)
      newMean = par
      limit = 0
      worst_fit = maximum(fitness)

      # Store population and selection means
     popMean = drop(population * weightsPop)
     muMean = newMean

     # Matrices for creating diffs
     diffs = zeros(N, lambda)
     x1sample = zeros(lambda)
     x2sample = zeros(lambda)

     chiN = sqrt(N)

     histNorm = 1 / sqrt(2)
     counterRepaired = 0

     stoptol = false

      while counteval < budget && !stoptol
        iter = iter + 1
        histHead = (histHead % histSize) + 1

        mu = floor(lambda / 2)
        weights = Base.log.(repeat([mu + 1],convert(Int64, mu))) - Base.log.(1:convert(Int64, mu))
        weights = weights / sum(weights)

        if log_Ft
           Ft_log = vcat(Ft_log, Ft)
        end
        if log_value
           value_log = vcat(value_log, fitness)
        end
        if log_mean
          mean_log = vcat(mean_log, fn_l(bounceBackBoundary2(newMean)))
        end
        if log_meanCord
          meanCords_log = vcat(meanCords_log, newMean)
        end
        if log_pop
          pop_log[:, :, iter] = population
        end
        if log_bestVal
           bestVal_log = vcat(bestVal_log, minimum(minimum(bestVal_log), minimum(fitness)))
        end
        if log_worstVal
           worstVal_log = vcat(worstVal_log, maximum(maximum(worstVal_log), maximum(fitness)))
        end
        if log_eigen
           eigen_log = vcat(eigen_log, reverse(sort(eigvals(cov(transpose(population))))))
        end

       # Select best 'mu' individuals of popu-lation
       selection = sortperm(vec(fitness))[1:convert(Int64,mu)]
       selectedPoints = population[:, selection]

       push!(history, selectedPoints.* histNorm./ Ft)

       # Calculate weighted mean of selected points
       oldMean = newMean
       newMean = drop(selectedPoints * weights)

       # Write to buffers
       muMean = newMean
       dMean[:, convert(Int64,histHead)] = (muMean - popMean)./ Ft

       step = (newMean - oldMean)./ Ft

       # Update Ft
       FtHistory[convert(Int64,histHead)] = Ft
       oldFt = Ft

      # Update parameters
      if  convert(Int64,histHead) == 1
        pc[:, convert(Int64,histHead)] = fill(0.0, N).*(1 - cp)./ sqrt(N) + step.*sqrt(mu * cp * (2 - cp))
      else
        pc[:, convert(Int64,histHead)] = pc[:, convert(Int64,histHead - 1)].*(1 - cp) + step.*sqrt(mu * cp * (2 - cp))
      end

      # Sample from history with uniform distribution
      if iter < histSize
        limit = histHead
      else
        limit = histSize
      end
      historySample = sample(1:limit, lambda, replace=true)
      historySample2 = sample(1:limit, lambda, replace=true)

      x1sample = sampleFromHistory(history, historySample, lambda)
      x2sample = sampleFromHistory(history, historySample, lambda)

      # Make diffs
      for i in 1:lambda
        x1 = history[convert(Int64,historySample[i])][:, x1sample[i]]
        x2 = history[convert(Int64,historySample[i])][:, x2sample[i]]

        diffs[:, i].= sqrt(cc) * ((x1 - x2) + dMean[:, convert(Int64,historySample[i])]).*randn(1) + pc[:, convert(Int64,historySample2[i])].*sqrt(1 - cc).* randn(1)[1]
      end

     # New population
     population = newMean.+ diffs.*Ft.+ reshape(tol * (1 - 2 / N^2)^(iter / 2) * rand(Normal(), length(diffs)), N, lambda)./ chiN
     population = deleteInfsNaNs(population)

     # Check constraints violations
     # Repair the individual if necessary
     populationTemp = population
     populationRepaired = apply(population, bounceBackBoundary2) #mapslices(bounceBackBoundary2, population, dims=[1])
     populationRepaired = reshape( hcat(populationRepaired...), size(populationTemp,1), size(populationTemp, 2))
     counterRepaired = 0

     for tt in 1:size(populationTemp,2)
       if any(populationTemp[:, tt] != populationRepaired[:, tt])
         counterRepaired = counterRepaired + 1
       end
     end

     if Lamarckism == true
        population = populationRepaired
     end

     popMean =  drop(population * weightsPop)

      # Evaluation
     fitness = fn_l(population)

     if Lamarckism == false
        fitnessNonLamarcian = fn_d(population, populationRepaired, fitness)
     end

     wb = argmin(vec(fitness))

     if fitness[wb] < best_fit[1]
        best_fit = fitness[wb]
        if Lamarckism == true
          best_par = population[:, wb]
        else
          best_par = populationRepaired[:, wb]
        end
     end

      # Check worst fit:
     ww = argmax(fitness)
     if fitness[ww] > worst_fit
        worst_fit = fitness[ww]
     end

      # Fitness with penalty for nonLamarcian approach
      if Lamarckism == false
        fitness = fitnessNonLamarcian
      end

      # Check if the middle point is the best found so far
      cumMean = 0.2 * newMean.+ 0.8 * cumMean
      cumMeanRepaired = bounceBackBoundary2(cumMean)

      fn_cum = fn_l(cumMeanRepaired)

      if fn_cum[1] < best_fit[1]
        best_fit = fn_cum
        best_par = cumMeanRepaired
      end

      if fitness[1] <= stopfitness
        msg = "Stop fitness reached."
        break
      end

    end


  end

    log = []

    if log_Ft
       log_Ft = Ft_log
     end
    if log_value
       log_value = value_log[1:iter, :]
     end
    if log_mean
       log_mean = mean_log[1:iter]
     end
    if log_meanCord
       log_meanCord = meanCords_log
     end
    if log_pop
       log_pop = pop_log[:, :, 1:iter]
     end
    if log_bestVal
       log_bestVal = bestVal_log
     end
    if log_worstVal
       log_worstVal = worstVal_log
     end
    if log_eigen
      log_eigen = eigen_log
    end

     res = [
       best_par,
       best_fit,
       convert(Int32, counteval),
       restart_number,
       ifelse(iter >= maxiter, 1, 0),
       msg,
       log,
       iter
     ]

   end
     return res

  end

function rnormFromMatrix(m)
  for i in 1:size(m, 1)
    for j in 1:size(m, 2)
      m[i,j]=randn(m[i, j])
    end
  end
end

function crossprod(x)
  if size(x, 2) == 1
    return sum(x.^2)
  end
  return transpose(x)*x
end

function rastrigin(x)
  d = length(x)
  sum = 10*d
  for i in 1:length(x)
    sum = sum + x[i]*x[i] - 10 * cos(2*pi*x[i])
  end
  return sum
end

function griewank(x)
  sum = 0
  p = 1
  for i in 1:length(x)
    p = p * cos(x[i]/sqrt(i))
    sum = sum + x[i]*x[i]
  end
  sum = sum/4000
  sum += 1
  sum = sum - p
  return sum
end
