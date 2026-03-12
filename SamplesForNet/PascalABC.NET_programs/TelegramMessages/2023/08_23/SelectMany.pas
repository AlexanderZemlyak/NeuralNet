begin
  var a := ArrGen(10,i->Random2(1,5));
  a.Println;
  var grs := a.GroupBy(x -> x[0]);
  grs.Println;
  grs.SelectMany(gr -> gr).Print;
end.