uses GraphWPF;

var a := 10.0;

procedure Draw;
begin
  SetMathematicCoords(-a,a);
  var pp := |Pnt(4,-3),Pnt(-3,-1),Pnt(3,3)|;
  Polygon(pp,ARGB(50,255,0,0));
end;

begin
  // Coordinate.Scale - сколько пикселей на 1 единицу
  // Coordinate.Origin := Pnt(x0,y0);
  // Еще надо масштабировать относительно некоторой точки на экране
  Window.Title := 'Событие колёсика мыши';
  Window.SetSize(640,480);
  Draw;
  OnMouseWheel := delta -> begin
    a -= 0.3 * Sign(delta);
    a := a.Clamp(0.1,100);
    Draw;
  end;
end.
