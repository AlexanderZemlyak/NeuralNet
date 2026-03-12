procedure p1 := Write(1);
procedure p2 := Write(2);
procedure ln := Writeln;

function f(n: integer): procedure := ()->Write(n);

begin
  (p1+p2+ln);
  (p2+p1+ln);
  (p1*10+p2*10+ln);
  ((p1+p2)*10+ln);
  
  (f(7)*5+f(3)*5+ln);
end.