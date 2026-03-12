begin
  var a := Arr('Hello', 'Bye', 'Solo');
  var d := a.ToDictionary(x -> x, x -> x.Length);
  Println(d);
  var d1 := a.ToDictionary(x -> x[1], x -> x);
  Println(d1);
  var dict := Arr(('Hello','привет'),('Bye','пока'),('One','один'));
  var d2 := dict.ToDictionary(x -> x[0], x -> x[1]);
  Println(d2);
end.