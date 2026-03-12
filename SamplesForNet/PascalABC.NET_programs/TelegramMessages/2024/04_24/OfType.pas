begin
  var a := Arr&<object>(1,2.5,3,'dghs','dg',3.5);
  a.OfType&<real>.Sum.Println;
  a.OfType&<integer>.Sum.Println;
  a.OfType&<string>.Select(s -> s.ToUpper).Println
end.