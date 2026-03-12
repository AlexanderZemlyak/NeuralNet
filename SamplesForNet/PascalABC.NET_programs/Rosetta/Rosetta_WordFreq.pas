// https://rosettacode.org/wiki/Word_frequency#PascalABC.NET

##
ReadAllText('135-0.txt').ToLower.MatchValues('\w+').EachCount
  .OrderByDescending(w -> w.Value).Take(10).PrintLines
 
