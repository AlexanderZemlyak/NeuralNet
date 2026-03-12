uses GraphWPF;

begin
  SetMathematicCoords(-1,10,-1);
  foreach var x in 1..9 do
  foreach var y in 1..9 do
    if (x,y) in (2..7).Cartesian(2..5) then
      Circle(x,y,0.1);
end.