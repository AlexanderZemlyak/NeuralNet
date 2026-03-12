uses TurtleWPF;

procedure Koch(sz: real; n: integer);
begin
  if n = 0 then
    Forw(sz)
  else begin
    Koch(sz/3,n-1); Turn(-60);
    Koch(sz/3,n-1); Turn(120);
    Koch(sz/3,n-1); Turn(-60);
    Koch(sz/3,n-1);
  end;
end;

begin
  SetSpeed(10);
  Turn(90);
  Forw(-300);
  Down;
  Koch(600,4);
end.