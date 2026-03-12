uses Timers,GraphWPF;

begin
  var t := new Timer(100);
  t.OnTimer := () -> Circle(Window.RandomPoint,30,RandomColor);
  t.Start;
end.