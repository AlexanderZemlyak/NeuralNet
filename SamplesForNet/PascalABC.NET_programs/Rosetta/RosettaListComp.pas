// https://rosettacode.org/wiki/List_comprehensions#PascalABC.NET

##
(1..20).CartesianPower(3).Where(\(x,y,z) -> (x*x + y*y = z*z) and (x < y)).PrintLines
