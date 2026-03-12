begin
  var x := 1.0;
  loop 310 do
  begin
    Println(x);
    x *= 10;
  end;
  
  Println('Самое большое вещественное: ', real.MaxValue);
end.
