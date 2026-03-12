type Cat = (Низкий, Средний, Высокий);

function Category(x: integer): Cat;
begin
  if x < 160 then
    Result := Низкий
  else if x < 185 then
    Result := Средний
  else Result := Высокий
end;

begin
  var a := ArrRandomInteger(50,140,200);
  a.GroupBy(x->Category(x)).EachCount.PrintLines;
end.