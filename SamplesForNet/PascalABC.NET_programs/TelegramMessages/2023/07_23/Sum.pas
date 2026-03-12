begin
  var sum := 0;
  loop 10 do
  begin
    var x := ReadInteger;
    sum += x
  end;
  Print(sum)
end.