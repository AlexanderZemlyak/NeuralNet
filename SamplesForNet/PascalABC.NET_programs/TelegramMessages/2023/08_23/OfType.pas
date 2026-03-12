begin
  var a := Arr&<object>(1,2.5,True,'z','Hello',7.0,4.3);
  a.Select(x -> (x,TypeName(x))).Println;
  a.OfType&<real>.Println;
  a.OfType&<real>.Sum.Println
end.