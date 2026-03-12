uses Drawman;

begin
  Task('a3');
  ToPoint(3,2); PenDown; OnVector(0,3); PenUp;
  ToPoint(4,4); PenDown; OnVector(5,-1); PenUp;
  
  ToPoint(2,1); PenDown; 
  OnVector(2,0); PenUp; OnVector(-1,0); PenDown; OnVector(3,0);
  PenUp;
  
  ToPoint(0,0);
end.