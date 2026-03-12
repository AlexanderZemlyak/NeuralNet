uses School;
begin
  var s := '12345';
  Print(Dec(s,8));
  Println(Dec(s,16));
  var i := 12345;
  Print(Bin(i));
  Print(Oct(i));
  Print(Hex(i));
  Print(ToBase(i,7));
end.