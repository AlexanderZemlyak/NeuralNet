uses GraphWPF;

procedure Redraw;
begin
  SetMathematicCoords(-20,20,true);
  var p1 := Pnt(10,10);
  var p2 := (-12,-3);
  Line(p1,p2);
  FillCircle(p1,0.2);
  FillCircle(p2,0.2);
end;

begin
  Brush.Color := Colors.Black;
  OnResize := Redraw;
end.