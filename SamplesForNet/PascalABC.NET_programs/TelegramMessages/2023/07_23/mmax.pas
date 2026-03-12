begin
  var mx := -integer.MaxValue;
  loop 10 do
  begin
    var x := ReadInteger;
    mx := Max(x,mx)
  end;
  Print(mx)
end.