begin
  var f: Text := OpenRead('flines.pas');
  foreach var x in f.Lines do
    Println(x);
  f.Close;
end.