##
var year := 2024;
var flag := False;
var n := 2;
Print(year,'=');
while year <> 1 do
  if year.Divs(n) then
  begin
    if flag then
      Print('*');      
    Print(n);
    Flag := True;
    year := year div n;
  end
  else n += 1;
