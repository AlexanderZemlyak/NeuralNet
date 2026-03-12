begin
  var a := MatrRandom(3,4,1,5);
  a.Println;
  foreach var (i,j) in a.Indices(x -> x = 1) do
    a[i,j] := 777;
  Println;
  a.Println;
end.