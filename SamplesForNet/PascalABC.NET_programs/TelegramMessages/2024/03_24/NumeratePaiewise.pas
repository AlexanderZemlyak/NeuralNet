##
var a := ArrRandomInteger(10);
a.Println;
foreach var (i,x) in a.Numerate do
  Print((i,x));
Println;
1.Step.ZipTuple(a).Println;

foreach var (x,y) in a.Pairwise do
  if x < y then
    Print((x,y));
Println;
a.ZipTuple(a.Skip(1)).Where(\(x,y) -> x < y).Println
    
  