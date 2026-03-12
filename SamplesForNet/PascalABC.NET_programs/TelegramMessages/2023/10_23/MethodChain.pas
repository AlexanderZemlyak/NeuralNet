begin
  var a := Arr(1..10);
  a.Println;
  a.Where(x -> x mod 2 = 0).Sum.Println;
  a[::2].Average.Println;
  a[5:].Min.Println;
  a[1::2].Where(x -> x mod 3 = 0).Count.Println;
  a[::-2].Select(x -> x + 100).Println.Sum.Println;
end.