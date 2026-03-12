##
// π = 3+4/(2·3·4)-4/(4·5·6)+4/(6·7·8)-4/(8·9·10)+4/(10·11·12)-4/(12·13·14)
var π := 0.0;
var sign := 1;
for var i := 1 to 1000 do
begin
  var x := i * 2;
  π += sign / x / (x+1) / (x+2);
  if sign = 1 then
    sign := -1
  else sign := 1
end;
π *= 4;
π += 3;
Println(π);
Println(Pi);