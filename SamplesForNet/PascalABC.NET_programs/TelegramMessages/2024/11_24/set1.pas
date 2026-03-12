begin
  var s: set of integer;
  var n := 10000000;
  for var i:=1 to n do
    s += [i];
  Print(MillisecondsDelta);
end.