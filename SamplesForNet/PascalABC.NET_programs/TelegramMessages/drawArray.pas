uses GraphWPF;

var 
  w := 30;  hh := 20;
  x0 := 20;  y0 := 20;
  x := x0;  y := y0;

procedure Draw(Self: array of integer); extensionmethod;
begin
  for var i:=0 to Self.Length-1 do
  begin
    Rectangle(x,y,w,w);
    DrawText(x,y,w,w,Self[i]);
    x += w;
  end;
  x += hh;
end;

procedure DrawLn(Self: array of integer); extensionmethod;
begin
  Self.Draw;
  y += w + hh;
  x := x0;
end;

begin
  Window.Title := 'Визуализация структур данных';
  Font.Size := 18;
  Arr(1,2,3).Draw;
  ArrRandom(10).Drawln;  
  ArrRandom(15).Drawln;  
  (1..17).ToArray.Drawln; 
end.
