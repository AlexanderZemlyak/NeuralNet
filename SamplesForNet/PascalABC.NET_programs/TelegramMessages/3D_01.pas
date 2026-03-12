uses Graph3D;

begin
  var b := Box(0, 0, 0, 3, 1, 1);
  var s := Sphere(1, 0.5, 0, 0.4);
  var s1 := s.Clone.MoveByY(-1);
  b.AddChild(s);
  b.AddChild(s1);
  b.Rotate(OrtZ, 45);
  b.Rotate(OrtY, -30);
  var a := Arrow(Origin, b.LocalAxisX * 5);
  b.AnimMoveByLocal(OrtX * 10, 2).Begin;
end.