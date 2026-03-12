begin
  var a := ArrRandomInteger(20,0,9);
  a.Println;
  a.Transform(x -> (if x < 5 then 5 else x));
  a.Println;
end.