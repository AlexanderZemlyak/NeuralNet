##
var a := ArrRandomInteger(20);

for var i:=0 to a.Length-2 do
   Swap(a[i],a[i+a[i:].IndexMin]);

a.Println;