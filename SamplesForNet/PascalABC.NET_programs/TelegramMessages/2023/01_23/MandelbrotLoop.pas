##
uses GraphWPF;

var m := new Color[800,600];
  
for var ix := 0 to 800 - 1 do
for var iy := 0 to 600 - 1 do
begin
  var z := Cplx(0,0);
  var n := 255;
  for var i:=0 to 255 do
  begin
    z := z * z + 0.0035 * Cplx(ix - 600,iy - 300);
    if z.Magnitude > 10 then
    begin  
      n := i;
      break;
    end;  
  end;
  m[ix,iy] := GrayColor(255-n);
end;

DrawPixels(0,0,m);

