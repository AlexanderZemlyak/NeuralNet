uses WPFObjects;

begin
  var sq := new SquareWPF(200,50,90,Colors.SkyBlue,1);
  var c := new CircleWPF(300,150,60,Colors.Green);
  Sleep(200);
  sq.MoveBy(20,0);
  Sleep(200);
  c.BorderWidth := 2;
  Sleep(200);
  c.ToBack;  
  Sleep(200);
  sq.Text := 'Hello';
end.