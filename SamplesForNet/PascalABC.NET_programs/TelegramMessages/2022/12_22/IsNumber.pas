uses GraphWPF;

begin
  Font.Name := 'Consolas';
  Font.Size := 20;
  var (x,y) := (1.0,1.0);
  var c0 := 10000;
  for var c := c0 to c0+160 do
  begin  
    TextOut(x,y,$'{Chr(c)} {c,-5}  ');
    x += 100;
    if x > 700 then (x,y) := (0,y+30);
  end;  
end.