begin
  var f := OpenRead('b.txt');
  var f1 := OpenWrite('c.txt');
  while not f.Eof do
    f1.Writeln(f.ReadString);
  f.Close;
  f1.Close;
end.