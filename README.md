# GA_berlin52_TSP
a GA algorithm that can reach a good result


The BASE Strategies:  
(1)Fitness:f(X)=X  
(2)Elitist strategy  
(3)Selection:Tournament selection  
  
  

More strategies:  
(1)We don’t think the individuals with bad performance can truly improve the final result, so we only use the half individuals which have better performance to do the GA each generation.  
(2)We think it is better that our populations are big, so that we can have more choices.  
(3)We think it is dangerous that MUTATION will destroy some good DNA,so we keep the MUTATION a low number.(In this experiment,we don't mutate at all).  


