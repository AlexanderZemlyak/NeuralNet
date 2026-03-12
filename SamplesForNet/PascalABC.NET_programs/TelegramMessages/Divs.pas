begin
  var x := 6;
  var b := (x mod 2 = 0) or (x mod 3 = 0);
  var b1 := x.Divs(2) or x.Divs(3);
  var b2 := x.DivsAny(2,3);
  Println(b,b1,b2);
end.