begin
  var a := [1,2,3];
  var b := a.Copy;
  b[0] := 777;
  Println(a,b);
  var m := Matr([1,2,3],[4,5,6]);
  var m1 := m.Copy;
  m1[0,0] := 888;
  Print(m,m1);
end.