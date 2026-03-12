type 
  Ученик = auto class
    Фамилия: string;
    Класс: integer;
  end;

begin
  var p1,p2: Ученик;
  p1 := new Ученик('Вася',13);
  p2 := new Ученик('Марина',14);

  Print(p1,p2);
end.

