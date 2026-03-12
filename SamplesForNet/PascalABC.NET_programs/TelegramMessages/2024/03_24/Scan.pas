begin
  var a := ArrRandom(10,1,9);
  a.Println.PartialSum.Println;
  a.Scan(min).Println;
  a.Scan(max).Println;
  a.Scan((x,y) -> x+y).Println;
  var s := 'abcdef';
  s.Scan('',(x,y) -> x+y).Println;
end.