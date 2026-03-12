begin
  var a := ArrRandomInteger(20,1,20);
  a.Println;
  var res := a.FirstOrDefault(x -> x > 17);
  Print(res);
  
  res := 0;
  foreach var x in a do
    if x > 17 then
    begin
      res := x;
      break;
    end;
  Print(res);      
end.