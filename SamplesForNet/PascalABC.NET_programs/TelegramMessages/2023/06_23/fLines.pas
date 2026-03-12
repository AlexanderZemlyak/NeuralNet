begin
  var f: Text := OpenRead('a.txt');
  foreach var x in f.Lines do
    Print(x);
  f.Close;
end.  