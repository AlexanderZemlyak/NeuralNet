uses WPFObjects;

begin
  var r := new RectangleWPF(100,100,50,70,Colors.Green);
  r.AnimRotate(90,2);
  var c := new CircleWPF(100,300,30,Colors.Red);
  c.AnimMoveBy(500,0,4);
  var t := new TextWPF(200,450,40,'PascalABC');
  t.AnimScale(2,2);
end.