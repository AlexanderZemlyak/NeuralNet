begin
  var a := 2.0; // exp(2.0)
  var n := 22;
  var x := 1.0;
  var s := x;
  for var i := 1 to n do
  begin
    x := x * a / i;
    s += x;
    Println($'{i} {s,17:f14}  {abs(s - exp(a)),0:E1}');
  end;
end.