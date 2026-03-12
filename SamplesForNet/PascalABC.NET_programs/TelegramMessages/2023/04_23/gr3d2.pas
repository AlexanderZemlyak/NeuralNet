uses Graph3D;

begin
  var bx := Box(0,0,0,6,4,1);
  bx.AddChild(CoordinateSystem(4));
  bx.Rotate(OrtY,30);
  var bbx := Box(0,0,2,2,1,1);
  bbx.AddChild(CoordinateSystem(3));
  bx.AddChild(bbx);
  var a1 := bbx.AnimRotate(OrtY,30);
  //(a1 + a2).Begin;
  a1.Begin;
  Sleep(1000);
  var a2 := bbx.AnimMoveByLocal(2,0,0);
  Print(bbx.LocalAxisX);
  a2.Begin;
end.