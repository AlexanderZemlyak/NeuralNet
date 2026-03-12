##
procedure SelectionSort(a: array of integer) :=
  for var i:=0 to a.Length-2 do
     Swap(a[i],a[i+a[i:].IndexMin]);

var a := ArrRandomInteger(20);
a.Println;
SelectionSort(a);
a.Println;