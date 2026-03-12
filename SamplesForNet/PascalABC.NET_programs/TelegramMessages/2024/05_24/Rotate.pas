uses Graph3D;

begin
  var c := Cylinder(-3,0,0,1,2); 
  c.Rotate(OrtY,90);
  c.Rotate(OrtZ,145); // поворачивает относительно локальной оси OZ, 
    // которая направлена по высоте цилиндра
  c.AnimRotateAtAbsolute(OrtZ,145,P3D(-2.5,0,0)).Begin; // поворачивает вокруг глобальной оси OZ
  Arrow(P3D(-2.5,0,0),V3D(0,0,5));
end.