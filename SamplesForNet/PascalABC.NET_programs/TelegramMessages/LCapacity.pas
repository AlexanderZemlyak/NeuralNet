begin
  var l := new List<integer>; 
  loop 20 do
  begin
    l.Add(1);
    Println($'Количество элементов = {l.Count,2}   Ёмкость списка = {l.Capacity,2}'); 
  end;
end.