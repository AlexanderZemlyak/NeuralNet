uses GraphWPF;

begin
  var (x1,x2) := (100.0,700.0);
  var (vx1,vx2) := (300,200); // пикселей в секунду
  OnDrawFrame := dt -> begin
    Circle(x1,250,50,Colors.Red);
    Circle(x2,400,50,Colors.Green);
    x1 += dt * vx1;
    x2 -= dt * vx2;
  end;
end.