begin
  var f := OpenRead('a.txt');
  while not f.Eof do
  begin
    var s := f.ReadString;
    Print(s.Trim);
  end;
  f.Close;
end.