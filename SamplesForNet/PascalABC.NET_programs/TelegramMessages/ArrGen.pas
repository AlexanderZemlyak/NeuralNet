begin
  var a := new integer[10];
  for var i:=0 to a.Length-1 do
    a[i] := i*2 + 1;
  a.Println;
  a := ArrGen(10,i → i*2 + 1);
  a.Println;
  a := ArrGen(10,1,i → i + 2);
  a.Println;
  a := (1..20).Step(2).ToArray;
  a.Println;
  a := Range(1,20,2).ToArray;
  a.Println;
end.