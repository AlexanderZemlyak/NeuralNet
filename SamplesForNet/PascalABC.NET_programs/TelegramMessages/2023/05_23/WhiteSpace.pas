begin
  for var c := #0 to #255 do
    if char.IsWhiteSpace(c) then
      Print(c.Code);
end.