begin
  var ss := ReadLines('ReadLines.pas');
  ss := ss.Select(s -> s[::-1]).Reverse;
  ss.PrintLines;
  WriteLines('a.txt',ss);
end.