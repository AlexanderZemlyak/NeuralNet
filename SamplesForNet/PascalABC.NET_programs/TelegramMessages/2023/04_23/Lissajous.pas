##
uses GraphWPF;

function Pos(t: real) := 
  Pnt(350*cos(4*t)+400, 250*cos(2.9*t + 2*Pi/3)+300);

Window.Title := 'Фигуры Лиссажу';
var t := 0.0;
var v := 0.2;
OnDrawFrame := dt -> begin
  t += dt * v;
  Circle(Pos(t),50,Colors.Green);
end;  
