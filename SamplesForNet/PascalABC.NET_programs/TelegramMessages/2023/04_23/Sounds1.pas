uses Sounds,GraphWPF,Controls;

begin
  var ss: Sound; 
  var b1 := Button(10,10,'Play',100);
  var b2 := Button(10,50,'Stop',100);
  b1.Click := procedure -> begin
    if ss<>nil then 
      exit;
    ss := new Sound('d:\www.mp3');
    ss.Play;
  end;
  b2.Click += procedure -> begin
    if ss<>nil then
      ss.Stop;
    ss := nil
  end;
end.
