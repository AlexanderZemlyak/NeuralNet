begin
  var (a,b) := Random2(1,10);
  Println(a,b);
  
  var min: integer;
  if a < b then
    min := a
  else min := b;
  
  var min1 := if a<b then a else b;
  
  Print(min,min1)
end.