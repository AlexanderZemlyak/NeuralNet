uses GraphWPF;

begin
  var p := Pnt(100,250);
  var p1 := Pnt(200,50);
  var p2 := Pnt(300,250);
  Arrow(p,p1);
  Arrow(p1,p2);
  Arrow(p,p2);
end.