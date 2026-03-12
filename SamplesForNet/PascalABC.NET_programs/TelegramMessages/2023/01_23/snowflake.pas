uses Turtle,GraphWPF;

const scale = 2.85;

procedure SnowFlake(a: real);
begin
  if a < 1 then exit;
  SnowFlake(a/scale);
  loop 6 do
  begin
    Down;
    Forw(a);
    SnowFlake(a/scale);
    Up;
    Turn(180);
    Forw(a);
    Turn(180);
    Turn(60);
  end;
end;

begin
  Window.Title := 'Снежинка';
  Pen.Color := Colors.Blue;
  SnowFlake(200);
end.