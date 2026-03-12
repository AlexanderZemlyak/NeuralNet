uses Graph3D;

begin
  var p1 := P3D(3,4,-2);  
  var p2 := P3D(-2,-3,4);
  Segment3D(p1,p2);
  var s := Sphere(p2,0.25);
  var tr: real -> Point3D := t -> t * p1 + (1-t) * p2;
  var seq := PartitionPoints(0,1,100).Select(tr);
  s.AnimMoveTrajectory(seq).AutoReverse.Forever.Begin;
end.