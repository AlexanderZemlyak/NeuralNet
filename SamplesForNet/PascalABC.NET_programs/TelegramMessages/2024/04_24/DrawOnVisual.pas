uses GraphWPF;

begin
  var v := CreateVisual;
  DrawOnVisual(v,dc → begin
    DrawRectangleDC(dc,100,100,200,120,Colors.Yellow,Colors.Black,2);
  end);
  OnMouseDown := (x,y,mb) -> DrawOnVisual(v,dc → begin
    DrawEllipseDC(dc,200,160,100,60,Colors.Yellow,Colors.Black,2);
  end);
end.