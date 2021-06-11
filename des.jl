# differential evolution strategy
using Distributions
using DataArrays

function test(par, fn, lower, upper, control = Dict())
   function controlParam(name, default)
     v = control[name]
     if isnothing(v)
         return(default)
     else
         return(v)
     end
   end

   function sampleFromHistory(history, historySample, lambda)
     ret = []
     for i in lambda
       ret = [ret, sample(1:length(history[historySample[i]]), 1)]
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
     pom = true
     for i in eachindex(x)
       if x[i] <= lower[i] || x[i] >= upper[i]
         pom = false
       end
     end
     if pom == true
       counteval = counteval + 1 # powinno byc <<- zamiast =
       return fn(x)
     else
       return prevfloat(typemax(Float64))
     end
   end

   function fn_l(P)
     if ndims(P) == 2 #if is.matrix()
        if counteval + size(P, 3) <= budget
          return mapslices(fn_, P, dims=[2])
        else
          ret = []
          budLeft = budget - counteval
          if budLeft > 0
            for i in budLeft
              ret = [ret, fn_(P[:,[i]])]
            end
          end
          return [ret, repeat(prevfloat(typemax(Float64)), size(P, 2) - budLeft)]
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
       for i in size(P,2)
         push!(repairedInd, P[:,[i]] != P_repaired[:,[i]])
       end
       P_fit = fitness
       P_pom = (P - P_repaired).^2
       vecDist = mapslices(sum, P_pom, dims=[1])
       P_fit[findall(repairedInd)] = worst_fit + vecDist[findall(repairedInd)]
       P_fit = deleteInfsNaNs(P_fit)
       return P_fit
     else
       P_fit = fitness
       if P != P_repaired
        P_pom = (P - P_repaired).^2
         P_fit = worst_fit + mapslices(sum, P_pom, dims=[1,2])
         P_fit = deleteInfsNaNs(P_fit)
       end
       return P_fit
     end
   end

   function bounceBackBoundary2(x)
     pom = true
     for i in eachindex(x)
       if x[i] <= lower[i] || x[i] >= upper[i]
         pom = false
       end
     end
     if pom == true
       return x
     elseif any(x.<lower)
       for i in findall(x.<lower)
         x[i] = lower[i] + abs(lower[i] - x[i]) % (upper[i] - lower[i])
       end
     elseif any(x.>upper)
       for i in findall(x.>upper)
         x[i] = upper[i] - abs(upper[i] - x[i]) % (upper[i] - lower[i])
       end
     end
     x = deleteInfsNaNs(x)
     return bounceBackBoundary2(x)
   end

   N = length(par)

   #dodac missing ewentualnie

   #############################
    ##  Algorithm parameters:  ##
    #############################
    Ft          <- controlParam("Ft", 1) ## Scaling factor of difference vectors (a variable!)
    initFt      <- controlParam("initFt", 1)


end
