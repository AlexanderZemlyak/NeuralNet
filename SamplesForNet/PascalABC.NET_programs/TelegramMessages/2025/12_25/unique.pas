begin
  var data := [1, 2, 3, 2, 4, 5, 3, 6, 7, 1, 8, 9, 4, 10];
  data.Println;
  
  var unique, multi: set of integer; 
  
  foreach var num in data do
  begin
    if num in multi then
      continue;
    if num in unique then
    begin
      unique -= num;
      multi += num;
    end 
    else unique += num;
  end;
  
  unique.Order.Println;
end.