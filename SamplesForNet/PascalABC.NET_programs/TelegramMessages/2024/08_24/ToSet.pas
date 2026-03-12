begin
  var x := 1668922;
  var y := 90192697672838;
  var s1 := x.ToString.ToSet;
  var s2 := y.ToString.ToSet;
  Println(x <= y);
end.