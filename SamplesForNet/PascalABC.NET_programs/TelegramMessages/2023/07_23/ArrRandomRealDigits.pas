begin
  var x := RandomReal(1,10,1);
  Println(x);
  var a := ArrRandomReal(10,1,10,digits := 1);
  a.Println;
  var q := SeqRandomReal(10,1,10,digits := 1);
  q.Println;
end.