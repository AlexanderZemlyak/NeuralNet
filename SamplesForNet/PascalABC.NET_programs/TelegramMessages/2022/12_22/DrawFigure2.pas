uses GraphWPF;
uses System.Windows.Media;
uses System.Windows;

procedure Fig(dc: DrawingContext);
begin
  var geo := new PathGeometry();
  var f := new PathFigure();
  geo.Figures.Add(f);
  var p0 := Pnt(30,0);
  var p2 := Pnt(15,30 * Sin(Pi / 3));

  f.StartPoint := p0;
  f.Segments.Add(new LineSegment(Pnt(0,0),true));
  f.Segments.Add(new LineSegment(p2,true));
  f.Segments.Add(new LineSegment(p0,true));
  
  var el := new EllipseGeometry(Pnt(0,0),8,8);
  var el1 := new EllipseGeometry(Pnt(22.5,15 * Sin(Pi / 3)),5,5);
  var el2 := new EllipseGeometry(Pnt(15,15 * Sin(Pi / 3)),2,2);
  var comb := new CombinedGeometry(GeometryCombineMode.Exclude,geo,el);
  var comb1 := new CombinedGeometry(GeometryCombineMode.Exclude,comb,el1);
  var comb2 := new CombinedGeometry(GeometryCombineMode.Exclude,comb1,el2);
  
  dc.DrawGeometry(ColorBrush(ARGB(64, 0, 255, 255)),ColorPen(Colors.Black),comb2);
end;

begin
  var a := 30;
  SetMathematicCoords(-2,40,-2,true);
  FastDraw(Fig);
end.