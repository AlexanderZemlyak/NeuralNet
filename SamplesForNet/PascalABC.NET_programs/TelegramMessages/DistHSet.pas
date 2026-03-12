##
uses Utils;

var n := 1000000;
var a := ArrRandomInteger(n,0,integer.MaxValue-1);

var tmp: integer;
Benchmark(() -> (tmp := a.Distinct.Count)).Println;
Benchmark(() -> (tmp := HSet(a).Count)).Println;