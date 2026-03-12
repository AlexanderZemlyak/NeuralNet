begin
  var Name: string;
  var Level: integer;
  case Name of
    'Алекс','Бонд': Level := 1;
    'Кей','Зед': Level := 2;
    'Игрек','Каппа': Level := 3;
    else: Level := 999;
  end;
  Print(Level);
end.