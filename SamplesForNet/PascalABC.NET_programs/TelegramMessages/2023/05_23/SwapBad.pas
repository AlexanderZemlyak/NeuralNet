begin
  var (a,b) := Random2(1,9);
  Println(a,b);
  a := a + b;
  b := a - b;
  a := a - b;
  Println(a,b);
end.