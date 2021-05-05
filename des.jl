# differential evolution strategy
1 using Distributions

function test(par, fn, lower, upper, control = HashTable([]))
   function controlParam(name, default)
     v = control[name]
     if is.null(v)
         return(default)
     else
         return(v)
     end
   end

end
