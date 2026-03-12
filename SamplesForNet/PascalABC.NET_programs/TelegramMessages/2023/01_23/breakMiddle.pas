begin
  var count := 0;
  while True do
  begin
    var x := ReadInteger;
    if x = 0 then
      break;
    if x mod 2 = 0 then
      count += 1;
  end;
  Print(count)
end.