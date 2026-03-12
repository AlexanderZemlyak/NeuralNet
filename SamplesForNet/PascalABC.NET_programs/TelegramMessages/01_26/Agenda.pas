begin
  var now := DateTime.Now;
  var t := now.TimeOfDay;

  var day := Arr(
    (7,30, 8,00, 'Подъём'),
    (8,00, 9,00, 'Завтрак'),
    (9,00,11,00, 'Глубокая работа'),
    (11,00,11,15,'Перерыв'),
    (11,15,13,00,'Учёба'),
    (13,00,14,00,'Обед'),
    (14,00,17,00,'Задачи'),
    (17,00,18,00,'Отдых')
  );

  foreach var d in day do
  begin
    var a := new System.TimeSpan(d.Item1,d.Item2,0);
    var b := new System.TimeSpan(d.Item3,d.Item4,0);
    var s := d.Item5;

    var mark := if (t >= a) and (t <= b) then '>>' else '  ';
    Println($'{mark} {a:hh:mm}-{b:hh:mm} {s}');
  end;
end.