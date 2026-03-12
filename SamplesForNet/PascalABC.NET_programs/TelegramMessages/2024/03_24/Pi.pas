##
// π = 4 * (1 - 1/3 + 1/5 - 1/7 ...)
var Pi1 := 0.0;
var sign := 1;
for var i := 1 to 1000000 step 2 do
begin
  if sign = 1 then
    Pi1 += 1/i
  else Pi1 -= 1/i;
  if sign = 1 then
    sign := -1
  else sign := 1
end;
Pi1 *= 4;
Println(Pi1);
Println(Pi);