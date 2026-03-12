begin
  {$IFDEF DEBUG}
  Print(1)
  {$ELSE}
  Print(2)
  {$ENDIF}
end.