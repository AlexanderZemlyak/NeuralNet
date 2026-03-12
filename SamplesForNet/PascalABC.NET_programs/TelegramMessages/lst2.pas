begin
  var x := 712456213;
  var l := new List<integer>;
  while x > 0 do
  begin
    l.Add(x mod 10);
    x := x div 10;
  end;
  l.Reverse;
  Print(l);
end.