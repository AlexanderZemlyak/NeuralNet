uses Graph3D;

begin
  Camera.LookDirection := V3D(-2,-18,-21);
  Sphere(Camera.Position,0.2);
  Arrow(Camera.Position,Camera.LookDirection);
  Camera.Position := Camera.Position - Camera.LookDirection;
  var b := Box(5.5,-4.5,2,1,1,4,Colors.Orange);
  b.AddChild(Cylinder(0,0,2,1,0.4,Colors.Red));
  b.AddChild(Cone(0,0,3,1.5,0.4,Colors.Gold));
  var Wall := Group(b);
  for var i:=1 to 11 do
    Wall.AddChild(b.Clone.MoveByX(-i));
  var Castle := Group(Wall);
  for var i:=1 to 3 do
    Castle.AddChild(Wall.Clone.RotateAt(OrtZ,90*i,Origin));
  Castle.AnimRotate(OrtZ,180,5).AutoReverse.Forever.Begin;
end.