uses ABCDatabases,GraphWPF;

procedure DrawHistogram(a: sequence of Ученик; x,y,w:real);
begin
  foreach var p in a do
  begin
    Rectangle(x,y,w,-p.Рост);
    DrawText(x,y,w,-p.Рост,p.Рост);
    TextOut(x,y+10,p.Фамилия,Alignment.LeftBottom,60);
    x += w
  end;
end;

begin
  var pupils := ЗаполнитьМассивУчеников;
  var girls := pupils.Where(p -> p.Пол = Жен);
  DrawHistogram(girls,50,200,25);
  DrawHistogram(girls.OrderBy(p -> p.Рост),50,500,25);
end.

