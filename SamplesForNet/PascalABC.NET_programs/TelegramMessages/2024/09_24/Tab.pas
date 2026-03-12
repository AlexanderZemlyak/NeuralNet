##
var (a,b,h) := (1.0,2.0,0.1);
foreach var x in Range(a,b,h) do
  Println($'{x,5:f2} {Sin(x),9:f4}');
