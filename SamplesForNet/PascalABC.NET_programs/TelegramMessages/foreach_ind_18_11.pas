##
var a := ArrRandomInteger(10);
a.Println;
foreach var x in a index i do
  if i.IsEven and x.IsEven then
    a[i] += 100;
a.Println;
  