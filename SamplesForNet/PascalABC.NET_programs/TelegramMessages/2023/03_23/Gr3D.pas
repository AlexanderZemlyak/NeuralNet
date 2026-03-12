uses Graph3D;

begin
  var w := 30;
  (1..5).Cartesian(3).ForEach(\(x,y,z) -> begin Text3D(x*w,y*w,z*w,''+Random(10),1) end)
end.