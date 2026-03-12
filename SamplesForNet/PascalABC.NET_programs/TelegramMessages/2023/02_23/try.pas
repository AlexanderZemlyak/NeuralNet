begin
  var n: integer;
  var InputError: boolean;
  repeat
    InputError := False;
    try
      n := ReadInteger;
    except
      InputError := True;
    end;
  until InputError = False;
  Print('Верный ввод:',n)
end.