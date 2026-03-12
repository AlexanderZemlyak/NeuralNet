##
uses School;

for var n := 1 to 100 do
begin
  var b := Bin(n);
  if b.CountOf('1').IsEven then
  begin  
    b += '0';
    b[:3] := '10';
  end
  else
  begin
    b += '1';
    b[:3] := '11';
  end;
  var r := Dec(b,2);
  if r > 40 then
  begin
    Print(n);
    exit
  end;
end;

