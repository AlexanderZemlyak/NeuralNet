procedure PrintFile(name: string);
begin
  if not FileExists('a1.pas') then
    raise new System.
end;

begin
  var n := 17;
  var fact := 1;
  for var i:=1 to n do
    fact *= i;
  Assert(fact > 0);
  
end.