uses GraphWPF;

begin
  Window.Title := 'С Новым годом!';
  Pie(400, 300, 250, -45, -135, Colors.Green);
  Pie(400, 200, 200, -45, -135, Colors.Green);
  Pie(400, 100, 150, -45, -135, Colors.Green);
  
  repeat
    Circle(300, 450, 25, RandomColor);
    Circle(450, 350, 20, RandomColor);
    Circle(360, 290, 25, RandomColor);
    Circle(480, 480, 25, RandomColor);
    Circle(350, 200, 20, RandomColor);
    Rectangle(350,400,50,30, Colors.LightSeaGreen);
    Arc(400,240,250,-55,-125,Colors.Yellow);
    Sleep(200);
  until False;
end.