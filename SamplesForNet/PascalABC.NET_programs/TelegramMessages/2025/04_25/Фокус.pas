##
'КОСУФ'
.CartesianPower(5)
.Numerate
.Where(t -> t[1].IsMatch('\b[^Ф]+\b'))
.Where(t -> t[1].CountOf('У') = 2)
.Last
.Print;