uses WPFObjects;

begin
  Window.Title := 'Перемещение объектов мышью';
  var sq := new SquareWPF(200,50,90,Colors.SkyBlue,1);
  var c := new CircleWPF(300,150,60,Colors.Green);
  var obj: ObjectWPF;
  OnMouseDown := (x,y,mb) -> begin
    obj := ObjectUnderPoint(x,y);
    if obj<>nil then
      obj.ToFront;
  end;
  OnMouseMove := (x,y,mb) -> begin
    if obj<>nil then
      obj.Center := (x,y);
  end;
  OnMouseUp := (x,y,mb) -> (obj := nil);  
end.