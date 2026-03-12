uses Robot;

procedure Solve();
begin
  if WallFromRight then
    Paint
  else begin
    Right;
    Solve;
    Left
  end;
end;

begin
  StandardField;
  Solve;
end.