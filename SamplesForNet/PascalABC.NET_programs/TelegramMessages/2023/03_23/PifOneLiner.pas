##
(1..9).Cartesian(2).Select(\(x,y)->$'{x}*{y} = {x*y,-2} ')
  .Batch(9).PrintLines(q -> q.JoinToString);