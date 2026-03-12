begin
  var a := ArrRandomReal(6);
  a.Println;
  var b := a.ConvertAll(x -> Round(x,2));
  b.Println;
  a.Select(x -> $'{x,4:f2}').Println
end.