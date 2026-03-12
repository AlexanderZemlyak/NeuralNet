uses PABCSystem,Coords,GraphWPF;

var (prevx, prevy) := (-1.0,-1.0);

begin
  Scale := 49;
  Origin := (7,5);
  OnMouseDown := (x,y,mb) -> begin
    if mb = 2 then 
    begin  
      (prevx, prevy) := (-1.0,-1.0);
      exit;
    end;  
    var p := fso.ScreenToReal(Pnt(x,y));
    var xx := Round(p.X);
    var yy := Round(p.Y);
    Coords.DrawCircle(xx,yy,0.05);
    if (prevx, prevy) <> (-1.0,-1.0) then
      DrawLine(prevx, prevy, xx, yy);
    (prevx, prevy) := (xx,yy);
  end;
end.
