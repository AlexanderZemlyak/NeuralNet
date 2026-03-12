##
function Mandelbrot(a: Complex): Complex;
begin
  var z := Cplx(0,0);
  for var i:=0 to 50 do
    z := z * z + a;
  Result := z;
end;

foreach var y in PartitionPoints(1,-1,40) do
begin
  foreach var x in PartitionPoints(-2,0.5,80) do
    if Abs(Mandelbrot(cplx(x,y))) < 2 then
      Write('*')
    else Write(' ');
    Writeln
end;  