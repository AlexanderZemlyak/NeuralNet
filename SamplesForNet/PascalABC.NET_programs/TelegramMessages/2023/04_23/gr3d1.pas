uses Graph3D;

begin
  var bx := Box(0,0,0,6,4,1);
  bx.AddChild(CoordinateSystem(4));
  bx.Rotate(OrtY,45);
  var bbx := Box(0,0,2,2,1,1);
  bbx.AddChild(CoordinateSystem(3));
  bx.AddChild(bbx);
  var a1 := bbx.AnimRotate(OrtY,45);
  a1.Begin;
  Sleep(1000);
  var a2 := bbx.AnimMoveByLocal(2,0,0);
  var a3 := bbx.AnimMoveByLocal(0,2,0);
  var a4 := bbx.AnimMoveByLocal(0,0,2);
  {a2.Begin;
  Sleep(1000);
  a3.Begin;
  Sleep(1000);
  a4.Begin;}
  (a2 + a3 + a4).Begin;
end.