begin
  var n := 291;
  var m := 35;
  Write(n div m,'.');
  loop 100 do
  begin
    if n = 0 then
      break;
    n := n mod m * 10;
    Write(n div m);
  end;  
end.