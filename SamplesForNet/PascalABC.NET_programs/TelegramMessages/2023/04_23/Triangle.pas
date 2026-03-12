uses GraphWPF;

begin
  SetMathematicCoords;
  var A := Pnt(-3,2);
  var B := Pnt(2,1);
  var C := Pnt(-2,-4);
  var arr := |A,B,C|;
  Polygon(arr,ARGB(100,255,228,156));
  Font.Size := 18;
  TextOut(A,A,Alignment.RightBottom);
  TextOut(B,B,Alignment.LeftBottom);
  TextOut(C,C);
end.