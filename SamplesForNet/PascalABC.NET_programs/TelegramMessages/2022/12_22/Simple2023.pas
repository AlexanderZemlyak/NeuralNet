##
var n := 2023;
var n0 := n;
var L := Lst(1);
var i:=2;
while i<=n do
begin  
  if n.Divs(i) then
  begin  
    L.Add(i);
    n := n div i;
  end
  else i += 1;
end;  
if L.Count = 1 then
  L.Add(n)
else L.RemoveAt(0);
  
Print(n0,'=',L.JoinToString(' * '));
    
  