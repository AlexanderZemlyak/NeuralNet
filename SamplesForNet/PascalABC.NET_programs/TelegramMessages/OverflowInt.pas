begin
  var x := 1;
  loop 11 do
  begin
    Println(x);
    x *= 10;
  end;
  
  Println('Самое большое целое: ', integer.MaxValue);
end.
