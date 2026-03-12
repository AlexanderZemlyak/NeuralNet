// https://rosettacode.org/wiki/Draw_a_rotating_cube#PascalABC.NET

uses Graph3D;

begin
  var Cube := Box(Origin,Sz3D(3,3,3),Colors.Green);
  Cube.AnimRotate(OrtZ,180).Forever.Begin
end.