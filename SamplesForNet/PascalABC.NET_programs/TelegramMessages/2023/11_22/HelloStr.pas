begin
  var s := 'Hello';
  var sb := new StringBuilder(s);
  for var i := 0 to sb.Length-1 do
    sb[i] := Succ(sb[i]);
  s := sb.ToString;
  Print(s);
end.