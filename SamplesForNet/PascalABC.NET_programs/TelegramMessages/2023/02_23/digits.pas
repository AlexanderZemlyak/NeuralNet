function Digits0(a: integer): sequence of integer;
begin
  Result := a.ToString.Select(c -> c.ToDigit);
end;

procedure Digits1(n: integer);
begin
  while n<>0 do
  begin
    Print(n mod 10);
    n := n div 10;
  end;
end;

function Digits2(n: integer): sequence of integer;
begin
  var L := new List<integer>;
  while n<>0 do
  begin
    L.Add(n mod 10);
    n := n div 10;
  end;
  Result := L;
end;

function Digits3(n: integer): sequence of integer;
begin
  while n<>0 do
  begin
    yield n mod 10;
    n := n div 10;
  end;
end;


function Digits4(a: integer): sequence of integer;
begin
  if a = 0 then exit;
  yield sequence Digits4(a div 10);
  yield a mod 10;
end;

begin
  var n := 234635681;
  Digits0(n).Println;
  Digits1(n);
  Println;
  Digits2(n).Println;
  Digits3(n).Println;
  Digits4(n).Println;
end.