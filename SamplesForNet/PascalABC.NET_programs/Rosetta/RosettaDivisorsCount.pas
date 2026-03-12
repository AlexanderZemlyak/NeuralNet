// https://rosettacode.org/wiki/Tau_function#PascalABC.NET

##
function DivisorsCount(n: integer) := Range(1,n).Count(i -> n.Divs(i));

Range(1,100).Select(n -> DivisorsCount(n)).Println