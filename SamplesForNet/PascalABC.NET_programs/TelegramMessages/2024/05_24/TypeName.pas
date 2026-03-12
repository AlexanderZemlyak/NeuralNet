begin
  var q := Arr(1..10);
  Println(TypeName(q.Select(x -> x + 0.5)));
  Println(TypeName(q.Where(x -> x.IsEven)));
  Println(TypeName(q.Pairwise));
  Println(TypeName(q.Select(x -> x + 0.5).Numerate));
  Println(TypeName(q.Cartesian(q)));
  Println(TypeName(q.Cartesian(q).ToList));
  Println(TypeName(q.EachCount));
end.