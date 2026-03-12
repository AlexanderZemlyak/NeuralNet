begin
  var f: Text;
  Assign(f, 'a.txt');
  Rewrite(f);
  Writeln(f,'Ancient Pascal');
  Close(f);
  
  var f1 := OpenWrite('b.txt');
  f1.Writeln('PascalABC.NET');
  f1.Close;
end.