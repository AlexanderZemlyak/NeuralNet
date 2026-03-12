begin
  var a := Arr((5,2),(2,7),(3,3));
  Println(a.MinBy(x -> x[0]+x[1]),a.IndexMinBy(x -> x[0]+x[1]));
  Println(a.MaxBy(x -> x[0]+x[1]),a.IndexMaxBy(x -> x[0]+x[1]));
end.