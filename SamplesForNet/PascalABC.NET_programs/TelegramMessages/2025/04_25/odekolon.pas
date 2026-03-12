##
'ОДЕКОЛОН'
.Permutations
.Where(x->not x.IsMatch('(.)\1'))
.ToSet
.Count
.Print;