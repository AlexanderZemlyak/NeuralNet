uses GraphWPF;

procedure DrawPoints(dc: DrawingContext);
begin
  var n := 100000;
  loop n do
    DrawEllipseDC(dc,Random(800),Random(600),7,7,ColorBrush(RandomColor),nil)
end;

begin
  FastDraw(DrawPoints);
end.