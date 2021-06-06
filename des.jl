# differential evolution strategy
using Distributions
using DataArrays

function test(par, fn, lower, upper, control = HashTable([]))
   function controlParam(name, default)
     v = control[name]
     if isempty(v)
         return(default)
     else
         return(v)
     end
   end

   function sampleFromHistory(history, historySample, lambda)

   end

   function deleteInfsNaNs(x)

   end
end
