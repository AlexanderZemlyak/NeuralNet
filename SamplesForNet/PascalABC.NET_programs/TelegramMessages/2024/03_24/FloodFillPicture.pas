uses GraphABC;

begin
  var pp := Picture.Create(300,300);
  pp.Circle(150,150,100);
  pp.FloodFill(150,150,Color.Yellow);
  pp.Save('d:\a.png');
end.