uses WPFObjects;

begin
  Window.Title := 'Перемещение игровго объекта анимацией на основе кадра';
  var Player := new CircleWPF(Window.Center, 20, Colors.Green);
  Player.Velocity := 200; // пикселей в секунду
  
  OnDrawFrame := dt -> begin
    Player.MoveTime(dt);
  end;
  
  OnKeyDown := k -> begin
    case k of
      Key.Left: Player.Direction := Direction.Left;
      Key.Right: Player.Direction := Direction.Right;
      Key.Up: Player.Direction := Direction.Up;
      Key.Down: Player.Direction := Direction.Down;
    end;
  end;
  
  OnKeyUp := k -> begin
    Player.Direction := (0, 0);
  end;
end.