type Vector = auto class
  x,y,z: real;
end;

function operator+(v1,v2: Vector): Vector;
begin
  Result := new Vector(v1.x +v2.x, v1.y +v2.y, v1.z +v2.z);
end;

begin
  var v1 := new Vector(1,2,3);
  var v2 := new Vector(4,5,6);
  Print(v1 + v2);
end.