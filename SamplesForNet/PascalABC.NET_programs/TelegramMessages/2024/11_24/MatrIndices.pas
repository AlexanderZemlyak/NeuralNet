begin
  var a := MatrRandom(3,4,0,2);
  a.Println(3);
  a.Indices(x -> x = 0).Println;
  a.Indices(x -> x = 0).ForEach(\(i,j) -> (a[i,j] := 99));
  a.Println(3);
end.