function F(a, limit, sm, pr: integer): integer;
begin
  if (a = limit) and (sm <= 1) and (pr <=1) then
    Result := 1
  else if (a > limit) or(sm > 1) or (pr > 1) then
    Result := 0
  else
    Result := F(a+1, limit, sm+1, 0) + F(a+2, limit, sm+1, 0) + 
      F(a*2, limit, 0, pr+1) + F(a*3, limit, 0, pr+1)
end;

begin
  Print(F(1, 22, 0, 0));
end.