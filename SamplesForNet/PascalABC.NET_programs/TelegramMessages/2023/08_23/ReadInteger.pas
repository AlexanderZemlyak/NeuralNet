begin
  var s := 0;
  loop 10 do
  begin
    var x := ReadInteger;
    s += x;
  end;
  Print('Сумма равна ',s);
  var c := ReadReal;
end.